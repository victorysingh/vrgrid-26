# Provenance: `checkpoints/frnet-semantickitti_seg.pth`

> **[!] READ THIS FIRST.** This file is an **independently sourced checkpoint from the FRNet
> authors' public release.** It is **NOT a reproduction of the original 4 Sep 2026 file** behind
> the recorded 90.3% / 65.2%. That file no longer exists anywhere: Shrestha does not have it, and
> a search of both repositories' full history (all 14 branches, no LFS, no file over 1 MB ever
> committed) shows it was never tracked. **The SHA-256 below is the ONLY provenance record that
> will ever exist for this file.** No checksum is published by the authors, and nothing can
> compare this file against the 4 Sep copy, which had no recorded hash either.

| field | value |
|---|---|
| **SHA-256** | **`09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`** |
| bytes | 40,327,676 (matches the served `Content-Length`) |
| format check | zip header `50 4b 03 04`, i.e. a PyTorch `.pth` archive, not an HTML page |
| source repository | https://github.com/Xiangxu-0103/FRNet (README: *"We provide the trained models for SemanticKITTI and nuScenes. The checkpoints can be downloaded from here"*) |
| source folder | https://drive.google.com/drive/folders/173ZIzO7HOSE2JQ7lz_Ikk4O85Mau68el |
| Drive file id | `1Ez-fpwu2WFCBw8usjwxUz6XQruw-cGU4` (resolved from the folder's public embedded view; the other file there, `1A2F8Rq56gsmD0u_bwpAWKYZLG6HRwda4`, is nuScenes and was not fetched) |
| download URL | `https://drive.usercontent.google.com/download?id=1Ez-fpwu2WFCBw8usjwxUz6XQruw-cGU4&export=download&confirm=t` |
| downloaded (UTC) | 2026-09-14T07:27:54Z → 07:28:00Z, `curl 8.12.1`, exit 0 |
| served `Content-Disposition` | `attachment; filename="frnet-semantickitti_seg.pth"` |
| served `Last-Modified` | Thu, 07 Dec 2023 02:42:16 GMT (matches the folder listing's 7 Dec 2023) |
| served `X-Goog-Hash` | `crc32c=Qaiqow==` (Google's transport checksum, not an authors' checksum) |
| SHA-256 computed | immediately after download, before the file was read by anything else |

The authors' README cites 73.3% mIoU on the SemanticKITTI **test** set. Our `frnet_eval.py` scores
the first 200 frames of **validation** seq 08 over the classes present there. The two numbers are
different measurements, and neither is expected to equal the other.

## Corroboration: an independent second download matches (2026-09-14)

JP downloaded the same file separately, through a browser, to
`C:\Users\JAIPREET SINGH\Downloads\frnet-semantickitti_seg.pth`. The file was created at 17:40:29
local time (+05:30), about 4.5 h after the curl download above. Checked read-only, nothing copied:

| check | result |
|---|---|
| SHA-256 | `09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`, **identical** |
| bytes | 40,327,676, identical |
| `cmp` against `checkpoints/frnet-semantickitti_seg.pth` | **identical byte for byte** |
| download origin (Windows `Zone.Identifier`) | `drive.usercontent.google.com`, the same file id `1Ez-fpwu2WFCBw8usjwxUz6XQruw-cGU4` (session token not recorded) |

**What this adds:** two independent downloads of the authors' public file, by different clients hours
apart, gave the same SHA-256. So the public file was stable across those hours, and the hash is a
sound identifier for it. **What it does not add:** any link to the original 4 Sep copy. This second
file was also downloaded today from the same public source, so the 4 Sep copy remains unrecoverable
and unverifiable.
