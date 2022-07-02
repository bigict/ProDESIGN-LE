# ProDESIGN-LE
ProDESIGN-LE, which is an accurate and efficient protein sequence design approach, designs a sequence using a transformer to iteratively maximize the overall fitness between residues and their local environments.

ProDESIGN-LE's corresponding article, _Accurate and efficient protein sequence design through learning concise local environment of residues_, is currently under review. The [bioRxiv verison](https://www.biorxiv.org/content/10.1101/2022.06.25.497605v1) is available currently, and a ProDESIGN-LE server is now available for non-commercial use.

## How to submit a pdb file for protein sequence prediction
Use the command below to send a pdb file to ProDESIGN-LE server for sequence design, substitute the "1812_a.pdb" (downlodable at this repo) with your own pdb file path. You will receive a designed sequence after ~30 seconds if your pdb contains 100 residues. Don't recommend submit a pdb with residues excedding 200, as ProDESIGN-LE is currently running on only 2 CPU cores.

If you receive error code `503' with error message 'Prediction failed', please make sure there is no break point in the backbone contained in your pdb file, and each residues should contain at least following atoms: N, Ca, and C.
Try to submit "1812_a.pdb" in this page as a test run, as this pdb file have already been tested.

```curl -X POST -H "Content-Type: text/plain" --data-binary @1812_a.pdb http://81.70.37.223:10088/predictions/prodesign```
