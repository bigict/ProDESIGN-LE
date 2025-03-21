# ProDESIGN-LE
ProDESIGN-LE, which is an accurate and efficient protein sequence design approach, designs a sequence using a transformer to iteratively maximize the overall fitness between residues and their local environments.

ProDESIGN-LE's corresponding article, _Accurate and efficient protein sequence design through learning concise local environment of residues_, is currently [published](https://academic.oup.com/bioinformatics/article/39/3/btad122/7077134).

## Basic Usage:

```python3 design.py best.pkl 0008.pdb -o . -c . -d cuda:0```

python3 design.py best.pkl \<PDB dir\> [-o output_path] [-c cache_path] [-d device] [-n number_designed_seqeunce]

## Options

Fix specific residues during design.
Add the flag ```--constrain <constrain.txt>```, where ```constrain.txt``` is a file structured as follows:
```
1 A
2 R
3 N
```
The first column represents the 1-indexed index of the residue, while the second column represents the one-letter residue symbol. Add as many rows as you like.

## Requirements:
1. pytorch  (checkout https://pytorch.org/get-started/locally/ for how to install)
2. einops  `pip install einops`
3. biopython  `pip install biopython`
