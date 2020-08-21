# snowy

Modelling and simulation of the operation of Snowy 2.0 in the National Electricity Market.
Analysis the influence on renewable generators and battery storage. 

The model can simulate both NSW and SA pricing. 
For NSW dispatch pricing use data found in 'full_data.csv'.
e.g. df = pd.read_csv('full_data.csv', names=colnames, nrows=n) 

For SA dispatch pricing use data found in 'full_data_SA.csv'.
e.g. df = pd.read_csv('full_data_SA.csv', names=colnames, nrows=n) 

For Pumped Hydro modelling use 'Snowy_2.0_Full_v3.py'.
For battery storage modelling use 'Snowy_2.0_Battery.pv'. 
