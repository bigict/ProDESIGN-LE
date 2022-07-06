# ProDESIGN-LE
ProDESIGN-LE, which is an accurate and efficient protein sequence design approach, designs a sequence using a transformer to iteratively maximize the overall fitness between residues and their local environments.

ProDESIGN-LE's corresponding article, _Accurate and efficient protein sequence design through learning concise local environment of residues_, is currently under review. The [bioRxiv verison](https://www.biorxiv.org/content/10.1101/2022.06.25.497605v1) is available currently, and a [ProDESIGN-LE server](http://81.70.37.223/) is now available for non-commercial use.


If you receive error code `503' with error message 'Prediction failed', please make sure there is no break point in the backbone contained in your pdb file, and each residues should contain at least following atoms: N, Ca, and C.
Try to submit "0738.pdb" in this page as a test run, as this pdb file have already been tested.
