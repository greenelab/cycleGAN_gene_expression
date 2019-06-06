# Gene expression transformation

**Background**
Chronic *Pseudomonas aeruginosa* infections are a serious problem to patients with Cystic fibrosis (CF).  These chronic infections lead to persistent inflammation which damages the lungs and ultimately results in respiratory failure, which is the leading cause of premature death in CF patients.  

Chronic *P. aeruginosa* infections tend to grow in biofilms.  Biofilms are characterized as sessile cells that stick to each other and to the surface forming a 3D structure.  In contrast, planktonic cells are free floating cells.  These biofilms are more representative of the CF lung environment, however transcriptional data grown in platonic cells are more readily available.  

**Question:**
Given the expression in planktonic cells can we predict the expression in biofilm cells?

**Hypothesis:**
A cycleGAN transformation of gene expression profiles from planktonic to biofilm states in matched conditions is improved compared to the average biofilm expression

**Dataset**
Dataset downloaded from ADAGE repository [ADAGE](https://github.com/greenelab/adage). This dataset is publically available *P. aeruginosa* gene expression data from ArrayExpress.  This collection contains ~100 microarray experiments under various conditions.  here are approximately ~1K samples with ~5K genes.  There are 976 planktonic and 138 biofilm samples.

