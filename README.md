# Search-based NMT
Neural machine translation (NMT) is an approach to machine translation.

# LFS
To run hebrew search-based model you need several big files (~200 MB), actually tables with nearest hebrew words. To get them you need install [git lfs](https://git-lfs.github.com/). For mac you can run:
```
brew install git-lfs
```
for archlinux you can run:
```
pacman -S git-lfs
```
Then run `git lfs install` and finally `git lfs pull` to download required lfs files.
