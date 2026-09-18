#!/usr/bin/env bash
# The T4 column: one g4dn.xlarge, scripted end to end. [Shrestha]
#
#   scripts/aws/t4.sh preflight      credentials, region, quota, price -- spends nothing
#   scripts/aws/t4.sh dryrun         EC2 DryRun of the exact launch: is a T4 allowed on this plan? free
#   scripts/aws/t4.sh budget         $50 monthly budget + email alert (runbook: FIRST)
#   scripts/aws/t4.sh stage labelled S3 bucket; upload sequences 00-10 + labels + poses (~50 GB)
#   scripts/aws/t4.sh stage all      ... or the full local dataset, all 22 sequences (~90 GB)
#   scripts/aws/t4.sh launch         key pair, SG (SSH from this IP only), instance, EIP
#   scripts/aws/t4.sh setup          clone, venv, Patchwork++, cupy, torch, data sync, tests
#   scripts/aws/t4.sh run            T4 measurements -> results/t4/ on the instance
#   scripts/aws/t4.sh fetch          copy results/t4 back (json/md/log only)
#   scripts/aws/t4.sh ssh            open a shell (tmux)
#   scripts/aws/t4.sh status | stop | start | cost
#
# MONEY: Shrestha's account has $100 of credits and nothing may be billed past
#   them. AWS has no hard spending cap, so this script is the cap:
#     - the budget tracks GROSS usage (credits excluded -- by default a budget
#       nets credits out, reads $0 and never alerts), with alerts at $10, $25,
#       $50 and a forecast alert at $50
#     - no Elastic IP (a public IPv4 bills even while the instance is stopped);
#       the IP is looked up per command instead
#     - a CloudWatch alarm STOPS the instance after 60 min under 5% CPU
#     - `run` arms an on-instance `shutdown -h` timer, the instance is set to
#       STOP (not terminate) on shutdown, and `run` stops it when it finishes
#     - disk sized to the data staged: 150 GB for `labelled`, 250 GB for `all`
#     - `cost` prints spend so far (Cost Explorer, gross of credits)
#
# Follows docs/gpu-lane/02-AWS-RUNBOOK.md: ap-south-1, g4dn.xlarge, Deep Learning
# Base OSS Nvidia Driver AMI (Ubuntu 22.04), 250 GB gp3, tagged Project=vrgrid,
# Owner=shrestha. There is deliberately NO terminate command: stop keeps the
# volume for ~$0.40/day, terminate destroys it, and that should be a decision
# made in the console, not a typo here.
#
# State (instance id, key path, bucket) lives in ~/.vrgrid-aws so the commands
# chain across shells.

set -euo pipefail

AWS="${AWS:-$HOME/.local/bin/aws}"
REGION="${VRGRID_AWS_REGION:-ap-south-1}"
TYPE="g4dn.xlarge"
# The disk is sized to what `stage` uploaded (see launch): every GB of gp3 bills
# ~$0.09/month for as long as the volume exists, stopped or not.
STATE="$HOME/.vrgrid-aws"
KEY_NAME="vrgrid-t4"
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
SG_NAME="vrgrid-t4-ssh"
TAGS="Key=Project,Value=vrgrid},{Key=Owner,Value=shrestha"
REPO="https://github.com/Stxtics03/vrgrid-26.git"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$STATE"
aws_() { "$AWS" --region "$REGION" "$@"; }
save() { echo "$2" > "$STATE/$1"; }
load() { cat "$STATE/$1" 2>/dev/null || true; }
die() { echo "t4.sh: $*" >&2; exit 1; }
ip() { aws_ ec2 describe-instances --instance-ids "$(load instance)" \
         --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }
ssh_() { ssh -i "$KEY_PATH" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$(ip)" "$@"; }

preflight() {
    "$AWS" sts get-caller-identity --output table || die "no AWS credentials -- run: $AWS configure"
    echo "region $REGION"
    aws_ ec2 describe-instance-type-offerings --location-type region \
        --filters "Name=instance-type,Values=$TYPE" --query 'InstanceTypeOfferings[].InstanceType' --output text \
        | grep -q "$TYPE" || die "$TYPE not offered in $REGION"
    # G and VT on-demand vCPU quota; a new account is often 0, and launch then fails
    local q
    q=$("$AWS" --region "$REGION" service-quotas get-service-quota --service-code ec2 \
        --quota-code L-DB2E81BA --query 'Quota.Value' --output text 2>/dev/null || echo "unknown")
    echo "G/VT on-demand vCPU quota: $q (g4dn.xlarge needs 4)"
    [[ "$q" == "unknown" ]] || awk "BEGIN{exit !($q >= 4)}" || \
        echo "!! quota below 4 -- request an increase in Service Quotas before launch"
    echo "AMI: $(ami)"
    echo "public IP of this machine: $(curl -s https://checkip.amazonaws.com)"
}

ami() {
    aws_ ec2 describe-images --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
                  "Name=state,Values=available" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text
}

budget() {
    local acct email n
    acct=$("$AWS" sts get-caller-identity --query Account --output text)
    email="${VRGRID_ALERT_EMAIL:?set VRGRID_ALERT_EMAIL to the address that gets the alert}"
    # Credits EXCLUDED: with them included the budget tracks the net bill,
    # which the credits hold at $0 right up until they run out.
    "$AWS" budgets create-budget --account-id "$acct" --budget '{
        "BudgetName":"vrgrid-t4","BudgetLimit":{"Amount":"50","Unit":"USD"},
        "TimeUnit":"MONTHLY","BudgetType":"COST",
        "CostTypes":{"IncludeCredit":false,"IncludeRefund":false,"IncludeTax":true,
                     "IncludeSubscription":true,"UseBlended":false}}'
    for n in 20 50 100; do     # $10, $25, $50 actual
        "$AWS" budgets create-notification --account-id "$acct" --budget-name vrgrid-t4 \
            --notification "NotificationType=ACTUAL,ComparisonOperator=GREATER_THAN,Threshold=$n,ThresholdType=PERCENTAGE" \
            --subscribers "SubscriptionType=EMAIL,Address=$email"
    done
    "$AWS" budgets create-notification --account-id "$acct" --budget-name vrgrid-t4 \
        --notification "NotificationType=FORECASTED,ComparisonOperator=GREATER_THAN,Threshold=100,ThresholdType=PERCENTAGE" \
        --subscribers "SubscriptionType=EMAIL,Address=$email"
    echo "budget vrgrid-t4: gross usage (credits excluded), email $email at \$10 / \$25 / \$50 and on a \$50 forecast"
}

cost() {
    local start end
    start=$(date -u +%Y-%m-01); end=$(date -u -d tomorrow +%F)
    # Cost Explorer bills $0.01 per request; this is one request.
    "$AWS" ce get-cost-and-usage --time-period "Start=$start,End=$end" --granularity MONTHLY \
        --metrics UnblendedCost --filter '{"Not":{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit"]}}}' \
        --group-by Type=DIMENSION,Key=SERVICE --output text
}

stage() {
    # shellcheck disable=SC1091
    source "$HERE/scripts/env.sh"
    local bucket
    bucket=$(load bucket)
    if [[ -z "$bucket" ]]; then
        bucket="vrgrid-data-$("$AWS" sts get-caller-identity --query Account --output text)"
        aws_ s3 mb "s3://$bucket" || true
        aws_ s3api put-bucket-tagging --bucket "$bucket" --tagging "TagSet=[{$TAGS}]"
        save bucket "$bucket"
    fi
    # `labelled` is every sequence the roadmap's AWS work reads: frnet_finetune
    # trains on 00-07, 09, 10 and frnet_eval scores 08. `all` adds 11-21, which
    # have no labels. The choice is an argument, never a default. `s3 sync`
    # resumes: re-run the same command after a dropped connection.
    local which="${1:-}" expect seqs
    python "$HERE/scripts/data_status.py" > /dev/null || die "local dataset incomplete; fix before staging"
    aws_ s3 sync "$VRGRID_DATA_ROOT/poses" "s3://$bucket/dataset/poses" --only-show-errors
    case "$which" in
        labelled) seqs=$(seq -w 0 10) ; expect=23201 ; save disk_gb 150 ;;
        all)      seqs=$(seq -w 0 21) ; expect=43552 ; save disk_gb 250 ;;
        *) die "stage needs 'labelled' (00-10, ~50 GB) or 'all' (00-21, ~90 GB)" ;;
    esac
    for q in $seqs; do
        aws_ s3 sync "$VRGRID_DATA_ROOT/sequences/$q" "s3://$bucket/dataset/sequences/$q" --only-show-errors
    done
    aws_ s3 cp "$VRGRID_FRNET_CHECKPOINT" "s3://$bucket/checkpoints/$(basename "$VRGRID_FRNET_CHECKPOINT")"
    local n
    n=$(aws_ s3 ls "s3://$bucket/dataset/sequences/" --recursive | grep -c '/velodyne/.*\.bin$' || true)
    echo "staged to s3://$bucket: $n velodyne scans (expect $expect for '$which')"
    [[ "$n" == "$expect" ]] || die "scan count mismatch -- re-run stage $which to resume"
}

dryrun() {
    # AWS answers "DryRunOperation" when the request WOULD succeed and an
    # authorization / unsupported error when it would not. Nothing is created.
    local out
    out=$(aws_ ec2 run-instances --dry-run --image-id "$(ami)" --instance-type "$TYPE" \
          --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}" \
          2>&1 || true)
    if grep -q "DryRunOperation" <<< "$out"; then
        echo "dry run: $TYPE launch would succeed in $REGION"
    else
        echo "$out" >&2
        die "dry run refused -- $TYPE cannot be launched on this account as it stands"
    fi
}

launch() {
    [[ -z "$(load instance)" ]] || die "instance $(load instance) already recorded; use start/status"
    if [[ ! -f "$KEY_PATH" ]]; then
        aws_ ec2 create-key-pair --key-name "$KEY_NAME" --query KeyMaterial --output text > "$KEY_PATH"
        chmod 400 "$KEY_PATH"
    fi
    local myip vpc sg id disk
    disk=$(load disk_gb)
    [[ -n "$disk" ]] || die "run stage first -- the disk is sized from what was staged"
    myip=$(curl -s https://checkip.amazonaws.com)
    vpc=$(aws_ ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
    sg=$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
         --query 'SecurityGroups[0].GroupId' --output text)
    if [[ "$sg" == "None" ]]; then
        sg=$(aws_ ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$vpc" \
             --description "vrgrid T4: SSH from one IP" --query GroupId --output text)
    fi
    aws_ ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
        --cidr "$myip/32" 2>/dev/null || true
    # No IAM role on the instance: `setup` hands it one-hour session credentials
    # for the S3 pull over SSH, so no long-lived key ever sits on the box.
    id=$(aws_ ec2 run-instances --image-id "$(ami)" --instance-type "$TYPE" \
         --key-name "$KEY_NAME" --security-group-ids "$sg" \
         --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$disk,VolumeType=gp3,DeleteOnTermination=true}" \
         --instance-initiated-shutdown-behavior stop \
         --tag-specifications "ResourceType=instance,Tags=[{$TAGS}]" "ResourceType=volume,Tags=[{$TAGS}]" \
         --query 'Instances[0].InstanceId' --output text)
    save instance "$id"
    echo "launched $id; waiting for running"
    aws_ ec2 wait instance-running --instance-ids "$id"
    # Idle stop: average CPU under 5% for 3 x 20 min. The action ARN needs no
    # IAM role. A forgotten instance then costs about an idle hour, not a month.
    aws_ cloudwatch put-metric-alarm --alarm-name "vrgrid-t4-idle-stop-$id" \
        --namespace AWS/EC2 --metric-name CPUUtilization --dimensions "Name=InstanceId,Value=$id" \
        --statistic Average --period 1200 --evaluation-periods 3 --threshold 5 \
        --comparison-operator LessThanThreshold \
        --alarm-actions "arn:aws:automate:$REGION:ec2:stop"
    aws_ ec2 wait instance-status-ok --instance-ids "$id"
    echo "ready: ssh -i $KEY_PATH ubuntu@$(ip)"
}

setup() {
    local bucket
    bucket=$(load bucket)
    [[ -n "$bucket" ]] || die "nothing staged; run stage first"
    # Short-lived credentials for the S3 pull, from this machine's session, so
    # the instance never holds long-lived keys.
    local creds
    creds=$("$AWS" sts get-session-token --duration-seconds 3600 \
            --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text)
    read -r AK SK ST <<< "$creds"
    # Power-off timer for the setup session, like `run` has: if this laptop
    # drops off mid-install, the instance still stops within 5 h.
    ssh_ "sudo shutdown -h +300 'vrgrid setup: 5 h cap'"
    ssh_ "bash -s" <<EOF
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv python3-dev cmake build-essential tmux unzip > /dev/null
if ! command -v aws > /dev/null; then
    curl -sS -o /tmp/awscli.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
    unzip -q -o /tmp/awscli.zip -d /tmp && sudo /tmp/aws/install > /dev/null
fi
AWS_ACCESS_KEY_ID=$AK AWS_SECRET_ACCESS_KEY=$SK AWS_SESSION_TOKEN=$ST \
    aws --region $REGION s3 sync s3://$bucket ~/assets --only-show-errors
[[ -d vrgrid-26 ]] || git clone -q $REPO
cd vrgrid-26 && git pull -q
python3 -m venv .venv && source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -e ".[dev]"
[[ -d ~/patchwork-plusplus ]] || git clone -q --depth 1 https://github.com/url-kaist/patchwork-plusplus.git ~/patchwork-plusplus
pip -q install ~/patchwork-plusplus/python
pip -q install cupy-cuda12x torch
python -c "import cupy, torch; print('cupy', cupy.__version__, cupy.cuda.runtime.getDeviceProperties(0)['name']); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
VRGRID_ASSETS=~/assets source scripts/env.sh
python scripts/data_status.py
python -m pytest -q -p no:cacheprovider 2>&1 | tail -3
EOF
}

run() {
    # Hard stop on the box itself: whatever happens to this laptop or the SSH
    # session, the instance powers down, and stops billing compute, in 4 h.
    ssh_ "sudo shutdown -h +240 'vrgrid run: 4 h cap'"
    ssh_ "bash -s" <<'EOF'
set -euo pipefail
cd vrgrid-26 && source .venv/bin/activate && VRGRID_ASSETS=~/assets source scripts/env.sh
mkdir -p results/t4
{ nvidia-smi; nvcc --version 2>/dev/null || true; lscpu | head -20; } > results/t4/host.log
python scripts/timing_table.py --seq 08 --frames 200 --device cpu  > results/t4/timing_cpu.log  2>&1
python scripts/timing_table.py --seq 08 --frames 200 --device cuda > results/t4/timing_cuda.log 2>&1
python scripts/gpu_parity.py   --seq 08 --frames 200               > results/t4/parity.log       2>&1
python scripts/vram_contention.py --seq 08 --frames 200 --pair-seconds 40 \
       --out results/t4/vram-contention.json                       > results/t4/vram_contention.log 2>&1
# Roadmap Day 1-2: fast-scatter verified on this machine, the pretrained numbers
# reproduced (90.3% / 65.2%), then the fine-tune -- trains on 00-07, 09, 10 and
# refuses 08 -- scored on 08 by the same script.
python scripts/frnet_fast_scatter.py                               > results/t4/fast_scatter_verify.log 2>&1
python scripts/frnet_eval.py --seq 08 --frames 200 --fast-scatter  > results/t4/frnet_eval.log   2>&1
python scripts/frnet_finetune.py --steps 600 --fast-scatter \
       --out ~/assets/checkpoints/frnet-finetuned-t4.pth           > results/t4/frnet_finetune.log 2>&1
python scripts/frnet_eval.py --seq 08 --frames 200 --fast-scatter \
       --checkpoint ~/assets/checkpoints/frnet-finetuned-t4.pth    > results/t4/frnet_eval_finetuned.log 2>&1
echo done
EOF
    # Results off the box first, then stop: compute is the part that bills by
    # the hour, and nothing should be left running once the logs are home.
    fetch
    echo "run finished -- stopping the instance now"
    aws_ ec2 stop-instances --instance-ids "$(load instance)" --output text
}

fetch() {
    mkdir -p "$HERE/docs/gpu-lane/t4"
    scp -i "$KEY_PATH" "ubuntu@$(ip):vrgrid-26/results/t4/*.{log,json}" "$HERE/docs/gpu-lane/t4/"
    ls -la "$HERE/docs/gpu-lane/t4"
}

status() {
    local id
    id=$(load instance)
    [[ -n "$id" ]] || { echo "no instance recorded"; return; }
    aws_ ec2 describe-instances --instance-ids "$id" \
        --query 'Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,PublicIpAddress]' --output text
}

case "${1:-}" in
    preflight) preflight ;;
    budget) budget ;;
    dryrun) dryrun ;;
    stage) stage "${2:-}" ;;
    launch) launch ;;
    setup) setup ;;
    run) run ;;
    fetch) fetch ;;
    ssh) ssh_ -t "tmux new -A -s vrgrid" ;;
    status) status ;;
    cost) cost ;;
    stop) aws_ ec2 stop-instances --instance-ids "$(load instance)" --output text ;;
    start) aws_ ec2 start-instances --instance-ids "$(load instance)" --output text
           aws_ ec2 wait instance-running --instance-ids "$(load instance)"; status ;;
    *) sed -n '2,17p' "$0"; exit 1 ;;
esac
