# Pick up here when AWS activates

Written 2026-09-17 18:30 UTC, while the account is still waiting to activate.
Everything below is either already done, or armed and waiting. Nothing in it
needs a decision from anyone to proceed — the decisions are listed separately
at the end, and they only matter if activation fails.

## Where it is stuck

The AWS account `317695557600` (root `takeindian@gmail.com`, created
2026-09-17T12:34Z) reads ACTIVE, has a default payment method (UPI AutoPay),
Cost Explorer is on — and EC2 still answers `OptInRequired`, S3 still answers
`NotSignedUp`. The console sends `/ec2` to
`signup.aws.amazon.com/billing/signup/incomplete`, whose own text says the
account has "either not finished registering, or your account is currently on
free plan", and that activation "might take up to 24 hours".

So it is one of two things, and we cannot tell which from outside:

1. activation is simply still running — the 24 h window closes
   **2026-09-18T12:34Z (18:04 IST)**; or
2. the **Free plan** itself withholds EC2 and S3, in which case waiting never
   fixes it and the account has to be upgraded to the Paid plan.

The tell is the clock: if EC2 is still refused after 18:04 IST on the 18th,
it is cause (2).

## What runs by itself

`scripts/aws/auto.sh labelled` is running in the background with
`VRGRID_ALERT_EMAIL=takeindian@gmail.com`. It polls EC2 and S3 every 10
minutes for up to 24 h, and the moment both answer it runs, in order and
stopping at the first failure:

    preflight → dryrun → stage labelled → launch → setup → run

`run` fetches the results back and stops the instance itself. If the poll
expires it exits 2 with `not active after 24 h -- contact AWS support` and
spends nothing. To drive it by hand instead, every step is a subcommand of
`scripts/aws/t4.sh` (see its header).

The API is the first signal, ahead of AWS's "Your AWS account is ready" email
— the poll sees activation before the inbox does.

## What comes back from the run

Into `docs/gpu-lane/t4/`, from one `g4dn.xlarge` (T4 16 GB) in ap-south-1:

- `host.log` — nvidia-smi, nvcc, lscpu, so the numbers have a machine attached
- `timing_cpu.log`, `timing_cuda.log` — seq 08, 200 frames: **the T4 column**
  of the frame-loop table, beside the laptop's 22.3 / 26.7 ms
- `parity.log` — `gpu_parity.py` on the T4: the CPU/CUDA map hashes must match
  there too, on a different card and a different driver
- `vram-contention.json` + log — VRAM attribution and the grid-vs-FRNet
  contention test repeated on 16 GB instead of the laptop's 8 GB, where the
  one-loop configuration was the one that missed 10 Hz
- `fast_scatter_verify.log`, `frnet_eval.log` — roadmap Day 1–2: fast scatter
  verified on this machine, then the pretrained FRNet reproduced against its
  published **90.3 % / 65.2 %**
- `frnet_finetune.log`, `frnet_eval_finetuned.log` — the 600-step fine-tune
  (trains on 00–07, 09, 10; refuses 08) scored on 08 by the same script

Day 9 of the roadmap then wants these numbers written into
`docs/research-log.md`, next to the laptop ones.

## The money, unchanged

$100 of credits, and nothing may be billed past them. Guardrails already in
place: budget `vrgrid-t4` at $50 of **gross** usage (credits excluded, or it
reads $0 forever) with email alerts at $10 / $25 / $50 and on a $50 forecast;
no Elastic IP; a CloudWatch alarm that stops the instance after 60 min under
5 % CPU; a `shutdown -h +240` armed on the box by `run`; stop-on-shutdown, not
terminate; disk sized to the staged data (150 GB for `labelled`).

Expected cost of the whole pass is single-digit dollars: a few hours of
g4dn.xlarge on demand, ~50 GB of S3 staging, and the gp3 volume at roughly
$0.40/day for as long as it is kept. `t4.sh cost` prints spend so far, gross
of credits. When the results are home, the volume and the bucket should be
deleted — that call is Shrestha's.

## Only if activation fails

Nothing here is automatic; both need Shrestha.

- **File a support case.** Basic support is free. Ask why EC2 and S3 are
  unsubscribed on an account that reads ACTIVE with a valid payment method.
- **Upgrade to the Paid plan.** Credits carry over and the alerts above stay
  live, but pay-as-you-go means spending past $100 lands on his card. This is
  his decision and his click, not mine.

## Laptop side, for reference

All of it is done, committed and pushed; CI green. D2/R3 ring boundary, the
band rebalance and the ring-3 fix on 08/09/10, the seq 07 and seq 09 regret
fixes, VRAM attribution and contention, R9's split/merge, traversability and
pyramid rows with the speedups (49.6 → 8.5 ms and 37.4 → 4.8 ms on CUDA, same
bits), R4 cell cost and stage attrition. Still open and *not* AWS-blocked:
N-3 (seq 00 ring-2 cross-look disagreement), N-4 (stale GitHub issue #6), and
the re-split of already-refined blocks flagged for Aakash.
