# Design of estimators for restoration of images degraded by haze using genetic programming, Swarm and Evolutionary Computation 44, pp. 49-63, (2019)

[J. E. Hernandez-Beltran](https://orcid.org/0000-0002-7043-3093), [V. H. Diaz-Ramirez](https://orcid.org/0000-0002-9331-1777), [Leonardo Trujillo](https://orcid.org/0000-0003-1812-5736), [Pierrick Legrand](https://www.math.u-bordeaux.fr/~plegra100p)

Paper Link: [DOI:10.1016/j.swevo.2018.11.008](https://doi.org/10.1016/j.swevo.2018.11.008)

Restoring hazy images is challenging since it must account for several physical factors that are related to the image formation process. Existing analytical methods can only provide partial solutions because they rely on assumptions that may not be valid in practice. This research presents an effective method for restoring hazy images based on genetic programming. Using basic mathematical operators several computer programs that estimate the medium transmission function of hazy scenes are automatically evolved. Afterwards, image restoration is performed using the estimated transmission function in a physics-based restoration model. The proposed estimators are optimized with respect to the mean-absolute-error. Thus, the effects of haze are effectively removed while minimizing overprocessing artifacts. The performance of the evolved GP estimators given in terms of objective metrics and a subjective visual criterion, is evaluated on synthetic and real-life hazy images. 

Please cite this paper as follows (Bibtex citation):

	@article{dehazingGP2019,		
	  title = "Design of estimators for restoration of images degraded by haze using genetic programming",
	  author = "J. E. Hernandez-Beltran and V. H. Diaz-Ramirez and L. Trujillo and P. Legrand",
	  journal = "Swarm and Evolutionary Computation",
	  pages = "49 - 63",
	  year = "2019",
	} 

## Example
<p align="center">
<img src="Images/img7.png" width="250px" height="200px"/>         <img src="Results/img7_outputPD.png" width="250px" height="200px"/>

## Prerequisites
Python 3

Install python packages: 
   opencv-contrib-python, numpy, scipy, scikit-image


## Usage
### To test the estimators GP-SD and GP-SN
	python GPStatistics.py   

### To test the estimators GP-PD and GP-PN
	python GPPixels.py   
	
NOTE: Read comments in the source file to select the estimators and modify parameters
