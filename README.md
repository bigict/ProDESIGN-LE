# ProDESIGN-LE
ProDESIGN-LE, which is an accurate and efficient protein sequence design approach, designs a sequence using a transformer to iteratively maximize the overall fitness between residues and their local environments.

ProDESIGN-LE's corresponding article, _Accurate and efficient protein sequence design through learning concise local environment of residues_, is currently under review. The [bioRxiv verison](https://www.biorxiv.org/content/10.1101/2022.06.25.497605v4) is available currently, and a [ProDESIGN-LE server](http://81.70.37.223/) is now available for non-commercial use.

## Usage:

```python3 design.py best.pkl 0008.pdb -o . -c . -d cuda:0```

python3 design.py best.pkl <PDB dir> [-o output_path] [-c cache_path] [-d device] [-n number_designed_seqeunce]

## Requirements:

TODO

## References
1. Carl Pabo. Molecular technology: designing proteins and peptides. Nature, 301(5897):200–200, 1983.
2. Aarón Terán, Aida Jaafar, Ana E Sánchez-Peláez, M Carmen Torralba, and Ángel Gutiérrez. Design and catalytic studies of structural and functional models of the catechol oxidase enzyme. Journal of Biological Inorganic Chemistry, 25(4):671–683, 2020.
3. Justin B Siegel, Alexandre Zanghellini, Helena M Lovick, Gert Kiss, Abigail R Lambert, Jennifer L St. Clair, Jasmine L Gallaher, Donald Hilvert, Michael H Gelb, Barry L Stoddard, et al. Computational design of an enzyme catalyst for a stereoselective bimolecular dielsalder reaction. Science, 329(5989):309–313, 2010.
4. Timothy A Whitehead, Aaron Chevalier, Yifan Song, Cyrille Dreyfus, Sarel J Fleishman, Cecilia De Mattos, Chris A Myers, Hetunandan Kamisetty, Patrick Blair, Ian A Wilson, et al. Optimization of affinity, specificity and function of designed influenza inhibitors using deep sequencing. Nature Biotechnology, 30(6):543–548, 2012.
5. Daniel-Adriano Silva, Shawn Yu, Umut Y Ulge, Jamie B Spangler, Kevin M Jude, Carlos Labão-Almeida, Lestat R Ali, Alfredo Quijano-Rubio, Mikel Ruterbusch, Isabel Leung, et al. De novo design of potent and selective mimics of IL-2 and IL-15. Nature, 565(7738):186–191, 2019.
6. Bruno E Correia, Yih-En Andrew Ban, Margaret A Holmes, Hengyu Xu, Katharine Ellingson, Zane Kraft, Chris Carrico, Erica Boni, D Noah Sather, Camille Zenobia, et al. Computational design of epitope-scaffolds allows induction of antibodies specific for a poorly immunogenic HIV vaccine epitope. Structure, 18(9):1116–1126, 2010.
7. Bruno E Correia, John T Bates, Rebecca J Loomis, Gretchen Baneyx, Chris Carrico, Joseph G Jardine, Peter Rupert, Colin Correnti, Oleksandr Kalyuzhniy, Vinayak Vittal, et al. Proof of principle for epitope-focused vaccine design. Nature, 507(7491):201–206, 2014.
8. Peng Xiong, Xiuhong Hu, Bin Huang, Jiahai Zhang, Quan Chen, and Haiyan Liu. Increasing the efficiency and accuracy of the ABACUS protein sequence design method. Bioinformatics, 36(1):136–144, 2020.
9. Namrata Anand, Raphael Eguchi, Irimpan I Mathews, Carla P Perez, Alexander Derry, Russ B Altman, and Po-Ssu Huang. Protein sequence design with a learned potential.
Nature Communications, 13(1):1–11, 2022.
10. Noelia Ferruz and Birte Höcker. Controllable protein design with language models. Nature Machine Intelligence, pages 1–12, 2022.
11. Sheng Chen, Zhe Sun, Lihua Lin, Zifeng Liu, Xun Liu, Yutian Chong, Yutong Lu, Huiying Zhao, and Yuedong Yang. To improve protein sequence profile prediction through image
captioning on pairwise residue distance map. Journal of Chemical Information and Modeling, 60(1):391–399, 2019.
12. Alexey Strokach, David Becerra, Carles Corbi-Verge, Albert Perez-Riba, and Philip M Kim. Fast and flexible protein design using deep graph neural networks. Cell Systems, 11(4): 402–411, 2020.
13. Brian Kuhlman, Gautam Dantas, Gregory C Ireton, Gabriele Varani, Barry L Stoddard, and David Baker. Design of a novel globular protein fold with atomic-level accuracy. Science, 302(5649):1364–1368, 2003.
14. Rebecca F Alford, Andrew Leaver-Fay, Jeliazko R Jeliazkov, Matthew J O’Meara, Frank P DiMaio, Hahnbeom Park, Maxim V Shapovalov, P Douglas Renfrew, Vikram K Mulligan, Kalli Kappel, et al. The Rosetta all-atom energy function for macromolecular modeling and design. Journal of Chemical Theory and Computation, 13(6):3031–3048, 2017.
15. James O’Connell, Zhixiu Li, Jack Hanson, Rhys Heffernan, James Lyons, Kuldip Paliwal, Abdollah Dehzangi, Yuedong Yang, and Yaoqi Zhou. SPIN2: Predicting sequence profiles
from protein structures using deep neural networks. Proteins: Structure, Function, and Bioinformatics, 86(6):629–633, 2018.
16. Yifei Qi and John ZH Zhang. DenseCPD: improving the accuracy of neural-network-based computational protein sequence design with DenseNet. Journal of Chemical Information
and Modeling, 60(3):1245–1252, 2020.
17. Yuan Zhang, Yang Chen, Chenran Wang, Chun-Chao Lo, Xiuwen Liu, Wei Wu, and Jinfeng Zhang. ProDCoNN: Protein design using a convolutional neural network. Proteins: Structure, Function, and Bioinformatics, 88(7):819–829, 2020.
18. Jianyi Yang, Ivan Anishchenko, Hahnbeom Park, Zhenling Peng, Sergey Ovchinnikov, and David Baker. Improved protein structure prediction using predicted interresidue orientations. Proceedings of the National Academy of Sciences, 117(3):1496–1503, 2020.
19. John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583–589,
2021.
20. Fusong Ju, Jianwei Zhu, Bin Shao, Lupeng Kong, Tie-Yan Liu, Wei-Mou Zheng, and Dongbo Bu. CopulaNet: Learning residue co-evolution directly from multiple sequence alignment for protein structure prediction. Nature Communications, 12(1):1–9, 2021.
21. Narayanan Eswar, David Eramian, Ben Webb, Min-Yi Shen, and Andrej Sali. Protein structure modeling with MODELLER. In Structural Proteomics, pages 145–159. Springer, 2008.
22. Patrick Conway, Michael D Tyka, Frank DiMaio, David E Konerding, and David Baker. Relaxation of backbone bond geometry improves protein energy landscape modeling. Protein
Science, 23(1):47–55, 2014.
23. Ivan Anishchenko, Samuel J Pellock, Tamuka M Chidyausiku, Theresa A Ramelot, Sergey Ovchinnikov, Jingzhou Hao, Khushboo Bafna, Christoffer Norn, Alex Kang, Asim K Bera,
et al. De novo protein design by deep network hallucination. Nature, 600(7889):547–552, 2021.
24. Ryan L Hayes and Charles L Brooks III. A strategy for proline and glycine mutations to proteins with alchemical free energy calculations. Journal of Computational Chemistry, 42(15):1088–1094, 2021.
25. Arieh Yaron, Fred Naider, and S Scharpe. Proline-dependent structural and biological properties of peptides and proteins. Critical Reviews in Biochemistry and Molecular Biology, 28(1):31–81, 1993.
26. Iain A Murray and William V Shaw. O-acetyltransferases for chloramphenicol and other natural products. Antimicrobial Agents and Chemotherapy, 41(1):1–6, 1997.
27. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural
Information Processing Systems, 30, 2017.
28. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
29. Helen Berman, Kim Henrick, and Haruki Nakamura. Announcing the worldwide protein data bank. Nature Structural & Molecular Biology, 10(12):980–980, 2003.
30. Anders O Magnusson, Anna Szekrenyi, Henk-Jan Joosten, James Finnigan, Simon Charnock, and Wolf-Dieter Fessner. nanoDSF as screening tool for enzyme libraries and
biotechnology development. The FEBS Journal, 286(1):184–204, 2019.
31. AJ Miles, Robert W Janes, and Bonnie A Wallace. Tools and methods for circular dichroism spectroscopy of proteins: A tutorial review. Chemical Society Reviews, 2021.
32. Andreas Hoffmann, Kerstin Grassl, Janine Gommert, Christian Schlesak, and Alexander Bepperling. Precise determination of protein extinction coefficients under native and denaturing conditions using SV-AUC. European Biophysics Journal, 47(7):761–768, 2018.
