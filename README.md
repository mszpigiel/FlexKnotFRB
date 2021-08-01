# FlexKnotFRB

These are the codes used for our paper "What it takes to measure reionization with fast radio bursts". If you would like to use any of these methods please feel free to contact me for more details and explanations, my email address can be found [here](https://www.ast.cam.ac.uk/people/Stefan.Heimersheim) or you can just open an issue on here GitHub :)

## Generation of the synthetic data set
The codes A and B are responsible for generating the data sets. As we are
using the star formation rate from Behroozi et al 2019, and the ionization
history from Kulkarni et al. 2019 you will need to add those data before
running the codes (as I did not want to include their data without
permission here), but you can also just use the generated catalogue
with the next codes. But otherwise use it like this
```bash
mkdir -p data/ # in case you did not have the folder
python3 A_create_perfect_data.py
python3 B_generate_synthetic_observations.py
```

## Running the Nested Sampling analysis
To run the Nested Sampling chains use `D_run.py` (`C_likelihood.py` is
just the likelihood implementation). You can use it like this:
```bash
python3 D_run.py Mockdata_phys_sfr_z10%_norm100_obs1.npy planckpriors monotonous flexknot2 eprior varyzend hostcalx12

#  Mockdata_phys_sfr_z10%_norm100_obs1.npy is the file with synthetic data being loaded
#  planckpriors indicates to use Planck constraints for Omega_m and Omega_b_over_h
#  flexknot2 indicates using the FlexKnot algorhtm, and the number 2 gives the
#            "effective" number of knots, or the degrees of freedom divided by 2.
#            So "1" refers to moving start+end point, "2" adds one fully movable knot.
#  The 'monotonous' keyword enables enforcing a monotonous ionization history.
#  eprior indicates to use the common z*exp(-z) formulation for the prior on the FRB source distribution,
#         instead of the true prior distribution which might be unknown.
#  varyzend indicates moving the start and endpoints, otherwise they are fixed to redshifts 5 and 30, respectively.
#  I use the argument including 'host' (e.g. hostcalx12) to decide the output filename.
```
The example python line will take the 100 FRB data set, and run a
Nested Samplung run with 3 FlexKnots (Start point, end point and one
fully movable point) and vary Omega_m and Omega_b_over_h within priors
approximately (i.e. assuming no correlation) corresponding to Planck limits.

## Analysing the Nested Samplung runs and generating posterior constraints
This will be added shortly
