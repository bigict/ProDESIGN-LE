# ProDESIGN-LE
ProDESIGN-LE, which is an accurate and efficient protein sequence design approach, designs a sequence using a transformer to iteratively maximize the overall fitness between residues and their local environments.

ProDESIGN-LE's corresponding article, _Accurate and efficient protein sequence design through learning concise local environment of residues_, is currently [published](https://academic.oup.com/bioinformatics/article/39/3/btad122/7077134), and a [ProDESIGN-LE server](http://81.70.37.223/) is now available for non-commercial use.

## Basic Usage:

```python3 design.py best.pkl 0008.pdb -o . -c . -d cuda:0```

python3 design.py best.pkl \<PDB dir\> [-o output_path] [-c cache_path] [-d device] [-n number_designed_seqeunce]

## Options

Fix specific residues during design.
Add the flag ```--constrain <constrain.txt>```, where ```constrain.txt``` is a file structured as follows:
```
1 A
2 B
3 C
```
The first column represents the 1-indexed index of the residue, while the second column represents the one-letter residue symbol. Add as many row as you like.

## Requirements:
1. pytorch
2. einops
3. biopython
