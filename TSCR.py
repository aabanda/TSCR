from functions.landmarkers import read_landmarkes_TSCR
from functions.landmarkers import compute_landmarkers_TSCR
from functions.subsamples import create_subsample_TSCR
from functions.recommendations import TSCR

#Example of how to run the TSCR suystem
db_name = 'ItalyPowerDemand'


# The dataset needs to be in .txt format
input_dir = "data/"  # location of .txt file
subsample_output_dir = "data/subsample" # location of the subsampled dataset
landmarker_output_dir = "data/landmarkers" # location of the landmarkers




#Create subsample
create_subsample_TSCR(input_dir, db_name, subsample_output_dir)


#Compute landmarkers
compute_landmarkers_TSCR(db_name, subsample_output_dir, landmarker_output_dir)


#Read landmarkers (accuracies and times)
landmarkers, computation_times = read_landmarkes_TSCR(db_name, landmarker_output_dir)


#Time Series Classifier Recommendation 
TSCR(landmarkers, metatarget="classifier_accuracies")
TSCR(landmarkers, metatarget="complete_ranking")
TSCR(landmarkers, metatarget="topm_rankings")
TSCR(landmarkers, metatarget="best_set")
TSCR(landmarkers, metatarget="best")


