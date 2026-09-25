# DJI Thermal SDK goes here (locally only)

Download the DJI Thermal SDK (v1.8 for Matrice 4T) from DJI and unpack it into this folder.
Everything in `sdk/` except this file is git-ignored — the SDK is under DJI's license and must not be committed.

Then:

```bash
export PVDELTA_DIRP_LIB="$(pwd)/sdk/tsdk-core/lib/linux/release_x64"
```
