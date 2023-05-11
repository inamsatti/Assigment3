# A few preliminaries

# usual things to use
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for erf():
import math
from sklearn.cluster import KMeans

# useful constant
DAYS_YR = 365.0

# Month labels
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# The weekly CO2 ppm file is linked on the page:  https://www.esrl.noaa.gov/gmd/ccgg/trends/data.html
# and also via the "Trends in CO2" link on: https://www.esrl.noaa.gov/gmd/dv/data.html

# Can read in the most recent file online
# (This works in a Kaggle notebook if Settings --> Internet is set to Active.)
##co2_file = "ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_weekly_mlo.txt"
# Also available at/as:
#     https://www.esrl.noaa.gov/gmd/webdata/ccgg/trends/co2/co2_weekly_mlo.txt
# The last day (Saturday) included in weekly measurements (six days after the most recent date tabulated.)
##co2_file_date = '29Apr2023'  

# Or use a saved version from the CO2 Mauna Loa Weekly dataset
co2_file_date = '29Apr2023'
co2_file = "co2_weekly_"+co2_file_date+".txt"
# Dates related to GT and major strikes
#
# GT school strike week 1, Friday 24 Aug 2018, day 236
yr_GTwk1 = 2018.0+(236.0-0.5)/DAYS_YR

# The first GRETA, 15 March 2019, day 74
yr_GRETA = 2019.0 + (74.0-0.5)/DAYS_YR
##print(yr_GRETA)    # 2019.20

# subsequent GRETA's:
# 3 May 2019, day 123 (more Canada, USA involvement)
yr_GRETA_mini = 2019.0 + (123.0-0.5)/DAYS_YR
# GRETA-2, 24 May 2019, day 144 
yr_GRETA_2 = 2019.0 + (144.0-0.5)/DAYS_YR
# The general global strikes:
# GRETA-General, 20[-27] September 2019, day 263 
yr_GRETA_General = 2019.0 + (263.0-0.5)/DAYS_YR
# The second general global strike, before COP 25:
# GRETA-General-2, 29 Nov 2019, day 333 
yr_GRETA_COP25 = 2019.0 + (333.0-0.5)/DAYS_YR
# GRETA: Global Day of Climate Action, 25 Sept. 2020
yr_GRETA_GDCA = 2020.0 + (269.0-0.5)/DAYS_YR
# GRETA: Global Climate Strike, 19 March 2021 #NoMoreEmptyPromises
yr_GRETA_NMEP = 2021.0 + (78.0-0.5)/DAYS_YR

# Other dates of note are in the code for the "weekly" or "future" plots, below;
# e.g., COP26 in Nov. 2021.

# Dates that select the "BAU" data to fit and how far to extrapolate

# Choose a starting time for fitting/analysis
yr_start = yr_GRETA - 10.0

# Choose to fit the data up until the first GRETA:
yr_fit = yr_GRETA

# and extrapolate to some further time
yr_end = yr_GRETA + 5.0

# Read the weekly CO2 data
def read_file(filepath):

    """
    Read CO2 data from a specified file path and return the resulting DataFrame.

    Args:
        filepath (str): The path to the file containing the CO2 data.

    Returns:
        pandas.core.frame.DataFrame: The DataFrame containing the CO2 data.
    """

    global df_co2
    df_co2 = pd.read_csv(filepath, sep='\s+', comment='#', header=None,
                usecols=[0,1,2,3,4,5,6,7],
                names=['year','mm','dd','time','CO2_ppm','days','CO2_1yr','CO2_10yr'])
    
    return df_co2

read_file(co2_file)

# Note: the decimal "time" in the file is the middle of the first day of the week;
#       add 3 days to it to be the middle of the week ...picky,picky...

df_co2['time'] = df_co2['time'] + 3.0/DAYS_YR

# Down-select to keep just data from the desired starting time
df_co2 = df_co2[df_co2['time'] > yr_start]


print("\nRead in the CO2 data in desired range:\n")
print(df_co2.head(6))
print( 6*"   . . .  ")
print(df_co2.tail(6))

# Are there any missed data points in the range?
df_gaps = df_co2[df_co2['CO2_ppm'] < 0.0]
# Patch them using the previous point and their 1 year ago values...
for indx in df_gaps.index:
    df_co2.loc[indx,'CO2_ppm'] = (df_co2.loc[indx-1,'CO2_ppm'] - df_co2.loc[indx-1,'CO2_1yr']) + \
        df_co2.loc[indx,'CO2_1yr']
print("\nPatched {} data points in this range.\n".format(len(df_gaps)))

# Augment the dataframe to go to the extrapolated time, yr_end.
# Calculate the number of weeks to add
last_index = df_co2.index[-1]
last_time = df_co2.loc[last_index, 'time']
week_yr = (7.0/DAYS_YR)
more_weeks = int((yr_end - last_time)/week_yr)
more_rows = []
# and create the augmented dataframe
df_co2_aug = df_co2.copy()
for iadd in range(1, more_weeks+1):
    new_time = last_time + iadd*week_yr
    # df_co2_aug = df_co2_aug.append(pd.DataFrame([{'year': int(new_time),
    #                                               'mm': 0,
    #                                               'dd': 0,
    #                                               'time': new_time}]), sort=False)
    df_co2_aug = pd.concat([df_co2_aug, pd.DataFrame([{'year': int(new_time),
                                                        'mm': int(0),
                                                        'dd': int(0),
                                                        'time': float(new_time)}])])

    # Reset the index, drop the old index, and replace df_co2 with the augmented version:
    df_co2 = df_co2_aug.reset_index().drop('index',axis=1)

    # Add a yr_phase column:
    df_co2['yr_phase'] = df_co2['time']-df_co2['time'].astype(int)
    
# Include GTweek numbers as well

# Find the index of the first school-strike week (it should be the week starting Sunday 19 Aug 2018).
# The "time" is mid-week so subtract some days from the Friday and use that week:
wk1_index = (df_co2.index[df_co2['time'] > (yr_GTwk1 - 4/DAYS_YR)])[0]

# Put GT-week numbers in the dataframe
df_co2['GTwk'] = 0
for wkind in range(wk1_index,len(df_co2)):
    df_co2.loc[wkind,'GTwk'] = 1 + df_co2.loc[wkind-1,'GTwk']
    
# GT-week-72 Friday is just into 2020, check it with:
##df_co2[df_co2['time'] > 2019.95].head(5)

# Show the dataframe columns around the start of the GT school strikes
df_co2[abs(df_co2['time'] - yr_GTwk1) < 30.0/DAYS_YR ]

# Set the font size for plots
plt.rcParams.update({'font.size': 12})

# A quick plot of the data values
df_co2.plot.scatter('time','CO2_ppm',figsize=(10,4))
plt.title("Mauna Loa Weekly CO2 Data")
#plt.savefig('1Maunalao.png')
plt.show()


# Monthly ONI (Oceanic Nino Index) from the displayed table on:
#  https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php 
# From Jan 2009 through... Feb 2022  (manually updated)
ONI_values = np.array([-0.8, -0.8, -0.6, -0.3, 0, 0.3, 0.5, 0.6, 0.7, 1, 1.4, 1.6,
1.5, 1.2, 0.8, 0.4, -0.2, -0.7, -1, -1.3, -1.6, -1.6, -1.6, -1.6,
-1.4, -1.2, -0.9, -0.7, -0.6, -0.4, -0.5, -0.6, -0.8, -1, -1.1, -1,
-0.9, -0.7, -0.6, -0.5, -0.3, 0, 0.2, 0.4, 0.4, 0.3, 0.1, -0.2,
-0.4, -0.4, -0.3, -0.3, -0.4, -0.4, -0.4, -0.3, -0.3, -0.2, -0.2, -0.3,
-0.4, -0.5, -0.3, 0, 0.2, 0.2, 0, 0.1, 0.2, 0.5, 0.6, 0.7,
0.5, 0.5, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.2, 2.4, 2.6, 2.6,
2.5, 2.1, 1.6, 0.9, 0.4, -0.1, -0.4, -0.5, -0.6, -0.7, -0.7, -0.6,
-0.3, -0.2, 0.1, 0.2, 0.3, 0.3, 0.1, -0.1, -0.4, -0.7, -0.8, -1,
-0.9, -0.9, -0.7, -0.5, -0.2, 0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.8,
0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.1, 0.2, 0.3, 0.5, 0.5,
0.5, 0.5, 0.4, 0.2, -0.1, -0.3, -0.4, -0.6, -0.9, -1.2, -1.3, -1.2,
-1, -0.9, -0.8, -0.7, -0.5, -0.4, -0.4, -0.5, -0.7, -0.8, -1.0, -1.0,
-1.0, -0.9])

# Assign a decimal year to each value as the center of the month
ONI_times = 2009.0 + (0.5 + np.array(range(len(ONI_values))))/12.0

print("ONI monthly data from 2009 to Oct 2021 should be 13*12-2 = 154; size is", len(ONI_values))
print("ONI start, end years are:", ONI_times[0], ONI_times[-1])

# # Add the ONI values to the dataframe ... or not 
# if False:
#     # Can include an offset (see results in fitting section below)
#     onioff = 0.0
#     ##onioff = 7.5/12.0
#     ##onioff = -6.0/12.0
#     # Interpolation will assign most recent ONI value to times greater than most recent time.
#     df_co2['ONI']= np.interp(df_co2.time.values, ONI_times+onioff, ONI_values)

#     # Plot the ONI values
#     # Set the font size for plots
#     plt.rcParams.update({'font.size': 12})
#     # A quick plot of the data values
#     df_co2.plot.scatter('time','ONI',figsize=(10,4),c='darkgreen')
#     # Add 0, +/- 0.5 lines
#     plt.plot([ONI_times[0], ONI_times[-1]],[0.5,0.5],c='red')
#     plt.plot([ONI_times[0], ONI_times[-1]],[0.,0.],c='black')
#     plt.plot([ONI_times[0], ONI_times[-1]],[-0.5,-0.5],c='blue')
#     plt.title("ONI Value (Oceanic Nino Index from NOAA)")
#     plt.show()


# Use linear regression from the scikit-learn machine learning collection:
from sklearn.linear_model import LinearRegression

# The above routine was chosen since I'd been recently steeped in machine learning things.
# A more typical (and function-flexible) approach would be to use least squares fitting,
# e.g., curve_fit from scipy.optimize, as used by Adams at:
# https://piphase.wordpress.com/2019/06/03/climate-data-fourier-analysis/

# Setup the usual ML X,y data sets, includes future weeks as well
X = pd.DataFrame(df_co2[['time','yr_phase']])
y = pd.DataFrame(df_co2['CO2_ppm'])

# Add trend and periodic features to use in fitting

# Polynomial time functions, degree 1 to 5
# expand around yr_fit, use just linear and quadratic
time_ref = yr_fit
X['time1'] = (X['time'] - time_ref)
X['time2'] = (X['time'] - time_ref)**2

# Yearly shape
X['sin_t'] = np.sin(2.0*np.pi*X['yr_phase'])
X['cos_t'] = np.cos(2.0*np.pi*X['yr_phase'])
X['sin_2t'] = np.sin(4.0*np.pi*X['yr_phase'])
X['cos_2t'] = np.cos(4.0*np.pi*X['yr_phase'])
X['sin_3t'] = np.sin(6.0*np.pi*X['yr_phase'])
X['cos_3t'] = np.cos(6.0*np.pi*X['yr_phase'])
X['sin_4t'] = np.sin(8.0*np.pi*X['yr_phase'])
X['cos_4t'] = np.cos(8.0*np.pi*X['yr_phase'])
X['sin_5t'] = np.sin(10.0*np.pi*X['yr_phase'])
X['cos_5t'] = np.cos(10.0*np.pi*X['yr_phase'])
# Remove the phase and the time columns (time1 replaces time in fitting)
X = X.drop(['yr_phase','time'],axis=1).copy()

# Include the ONI value?
# Including it can improve somewhat the fit if an offset is used,
# but it does not qualitatively change (or explain) the post-GRETA decreasing trend,
# so comment out these 3 lines to leave the ONI out of the model.
##print("Including ONI, ONI^2 with offset =",12*onioff,"months.")
##X['ONI'] = df_co2['ONI']
##X['ONI2'] = df_co2['ONI']**2
# Including the ONI gives:
#  onioff (months)   R^2    sigma (ppm)   fit coef.s:
#          -12.0    0.9945  0.535
#           -9.0    0.9949  0.517
#           -6.0    0.9951  0.504       -0.25055147  0.01106232
#           -3.0    0.9949  0.517
#            0.0    0.9943  0.544       -0.04158918 -0.00749099
#            3.0    0.9946  0.531
#            6.0    0.9949  0.513
#            7.5    0.9950  0.512        0.18064688  0.03607943
#            9.0    0.9949  0.513
#           12.0    0.9948  0.521
#           15.0    0.9946  0.531
#           18.0    0.9947  0.528
#
# without ONI       0.9943  0.546
# Different shifts of note:
# -6.0 aligns the ~2016 ONI peak with a ~2015 CO2 residual low
#  7.5 aligns the ~2016 ONI peak with a 2016+ CO2 residual peak
#  0.0 has small coef.s and makes little change to the fit

# Use simple linear regression
model = LinearRegression()

# Do the fitting on a limited time-range of the data
# Get the data to use for fitting, up to yr_fit:
fit_rows = df_co2['time'] < yr_fit
Xfit = X[fit_rows].copy()
yfit = y[fit_rows].copy()

# and do the fit
fit_model = model.fit(Xfit, yfit)

# The R^2 is very close to 1.00:
score = fit_model.score(Xfit, yfit)
print("The fit has R^2 = {:.4f}".format(score))

# Note: previous to Feb. 2021 this value was R^2 = 0.9942

# Look at the fit features
##Xfit

# Put the model values in the original dataframe:
df_co2['BAU_ppm'] = fit_model.predict(X)

# along with residuals
df_co2['CO2_resid'] = df_co2['CO2_ppm'] - df_co2['BAU_ppm']

# and the model residual, i.e., zeros (for plotting)
df_co2['BAU_resid'] = 0.0

# Get the coefficients and intercept
cols = X.columns
coeffs = fit_model.coef_
intercept = fit_model.intercept_[0]
linear_coeff = coeffs[0,0]
ppmyryr_coeff = 2.0*coeffs[0,1]

# Show some useful values
print("\nThe model coefficients at {:.2f} are:\n".format(time_ref))
print("  Intercept:  {:.2f}".format(intercept))
print("     Linear:  {:.3f} ppm/yr".format(linear_coeff))
# multiply the quadratic coef by 2 to get the "acceleration": 
print(" dLinear/dt:  {:.3f} ppm/yr/yr".format(ppmyryr_coeff))

# The peak value in May 2020
print("  2020 Peak:  {:.2f}".format(df_co2.loc[(df_co2['time'] < 2020.50), "BAU_ppm"].max()))


# List the periodic coefficients for reference.
print("\nThe periodic sin,cos coeff.s are:\n", list(cols[2:]), "\n", coeffs[0,2:])


# Note: previous to Feb. 2021 some values were: 2.702, 0.061, 417.39.

# Create Data and Model columns with just secular or just periodic components

# Evaluate the model without the periodic terms:
df_co2['BAU_secular'] = intercept
for tpow in [1, 2]:
    df_co2['BAU_secular'] += X['time'+str(tpow)]*coeffs[0,tpow-1]
    
# Just the periodic part of the model, i.e., subtracting off the BAU_secular
df_co2['BAU_periodic'] = df_co2['BAU_ppm'] - df_co2['BAU_secular']


# Data: Create the secular ("deseasoned") Data, i.e., without periodic terms
df_co2['CO2_secular'] = df_co2['BAU_secular'] + df_co2['CO2_resid']

# Data: and the periodic Data without the secular terms:
df_co2['CO2_periodic'] = df_co2['BAU_periodic'] + df_co2['CO2_resid']

# Variation of the model parameters with the number of years fit...
#          Prediction at 2020 peak is in any case between 416 - 418
#  3:  410.61  2.93 ppm/yr.  0.25 ppm/yr/yr  0.9702  2020: 417.87  <-- upper
#  5:  410.44  2.26 ppm/yr. -0.06 ppm/yr/yr  0.9813  2020: 416.56  <-- lower
#  7:  410.64  2.55 ppm/yr.  0.01 ppm/yr/yr  0.9896  2020: 417.13
# 10:  410.79  2.70 ppm/yr.  0.03 ppm/yr/yr  0.9942  2020: 417.39  <-- use this as nominal
# 13:  410.88  2.77 ppm/yr.  0.04 ppm/yr/yr  0.9963  2020: 417.52
# 15:  410.74  2.69 ppm/yr.  0.03 ppm/yr/yr  0.9970  2020: 417.26
# 20:  410.40  2.53 ppm/yr.  0.02 ppm/yr/yr  0.9979  2020: 416.59
# 30:  410.57  2.57 ppm/yr.  0.02 ppm/yr/yr  0.9986  2020: 416.80
# 40:  409.93  2.43 ppm/yr.  0.02 ppm/yr/yr  0.9987  2020: 415.95

# Very approximate range of the model around the nominal 10-year fit:
model_std = 0.50

# Calculate the standard deviation of the fit residuals
df_stats = df_co2[fit_rows].describe()
resid_std = df_stats.loc['std','CO2_resid']
# Look at the distribution of the weekly residuals from the fit:
df_co2[fit_rows].hist(['CO2_resid'], bins='auto', figsize=(8,5))
plt.title("CO2 Model Fit Residuals ({}--{}): sigma = {:.3f} ppm".format(int(yr_start), int(yr_fit), resid_std))
#plt.savefig("2CO2Model.png")
# plt.show()

# Note: previous to Feb. 2021 this value was sigma = 0.548

# Calculate the periodic component for each day of the year
# Output these in case people want to correct Daily values for seasonal variation.
days_in_month =[31,28,31, 30,31,30, 31,31,30, 31,30,31]
day_num = 0
day_names = []
corrects = []
for imonth, month in enumerate(months):
    for iday in range(days_in_month[imonth]):
        # save the month day strings:
        day_names.append(month+' '+str(iday+1))
        day_num += 1
        # phase angle goes 0 to 2pi over the year,
        # include minus 0.5 to be at the middle of the day.
        phase = (day_num-0.5)/365.0
        # use the 5x2 periodic coef.s, coeffs[0,2:], to calculate the correction
        correct = 0.0
        for ifreq in [1,2,3,4,5]:
            angle = 2.0*np.pi*ifreq*phase
            correct += coeffs[0,2*ifreq]*np.sin(angle) + coeffs[0,2*ifreq+1]*np.cos(angle)
        # The amount we add on to do the correction is -1 times the periodic component:
        corrects.append(-1.0*correct)
        ##print(month, iday+1, phase, correct)

# Put them in a data frame
deseas = pd.DataFrame(day_names,index=None,columns=['day'])
deseas['add_this_to_ppm'] = corrects
# and write them out using 2 decimal digits accuracy
deseas.to_csv("CO2afterGRETA_seasonal_correction.csv",index=False,
             float_format='%.2f')

# Quick look at the correction values which are to be added on to correct of seasonal variation.
# Note: these are the negative of the Seasonal Component seen in the deseasoned insert plot further below.
plt.plot(corrects)
#plt.savefig("3corrections.png")
# plt.show()

# The IPCC Special Report "Global Warming of 1.5 C" suggests emissions paths with
# "no or limited overshoot" will have 2030 emissions ~45% below 2010 levels.

# The ppm/yr/yr (curvature) parameter is changed from its BAU value (~ +0.061) and
# set to a constant negative value to model the expected-desired future CO2 values.
# With this constant "deceleration", the model gives a parabolic CO2 concentration curve,
# i.e., a linear change in CO2 rate with time.
# The ppm/yr/yr is simply calculated by specifying:
#    i) yr_reduced = the year when the ppm/yr is to be reduced and
#   ii) reduced_fraction = the factor by which it is multiplied at that year.
# With "linear_coeff" the starting ppm/yr value (i.e., 2.701 at yr_fit),
# the desired ppm/yr/yr is then:
#    new_ppmyryr = (reduced_fraction - 1)*linear_coeff/(yr_reduced - yr_fit)
#
# [Note: In v51 and earlier versions, two "schemes" to model the expected CO2
#  change were described here.
#  In version v52 and following, only the correct "Scheme 2" is described.]
#
#
# How will CO2 change when emissions are following IPCC SR15?
#
# I used the BernSCM simulator to make a very simple model of emissions which have the
# IPCC "-45% of 2010 in 2030" level in the forcing function (for "fossil_CO2_em"),
# specifically:
#    2010 emissions: 8.46 GtC/yr
#    2020 emissions: 9.40
#    2025 emissions: 6.63
#    2030 emissions: 4.653    <-- This is 0.55 of the 2010 8.46 value.
# The *emissions* reduction per year from 2020 to 2025 is about -6.7% per year.
#
# What is the predicted ppm change in this case?
# The BernSCM output gives these ppm/yr values at the dates:
#          ppm/yr
#    2020  2.71
#    2025  1.27
#    2030  0.39 
# Focusing on the near term, from 2020 to 2025,
# the ppm/yr changes by a factor of 1.27/2.71 = 0.469, or an exponential factor of 0.859 per year.
# The *CO2* change is about -14.1 %/yr, roughly twice the emissions change.

# So to get the expected-desired ppm/yr/yr, use parameters:
yr_reduced = yr_GRETA + 5.0   # reduction in 5 years
reduced_fraction = 0.859**5   # reduced x0.469 in five years

# Use these values to calculate the needed ppm/yr/yr.
# Note "linear_coeff" is the starting ppm/yr value ~ 2.701. 
new_ppmyryr = (reduced_fraction - 1)*linear_coeff/(yr_reduced - yr_fit)

print("\nSetting dLinear/dt to {:.3f} ppm/yr/yr ".format(new_ppmyryr) +
     "to decrease the ppm/yr rate to {:.1f}% of its value at {:.2f}\n".format(
         100.0*reduced_fraction, yr_reduced))

# Because the model is quadratic there is a *constant* absolute change
# in the ppm/yr rate from year to year; this is the -0.288 value of ppm/yr/yr.
# This means that the *fractional* change in ppm/yr is not constant for this model;
# this the case in the IPCC SR 1.5 where pathways have their emissions from
# 2020 to 2030 decreasing along a straight line, rather than an exponential.
print("The percent change in ppm/yr varies with the year, some examples:")
print("= {:.2f}% in first year,".format(100.0*new_ppmyryr/linear_coeff))
print("= {:.2f}% in fifth year.".format(100.0*new_ppmyryr/(linear_coeff+4.0*new_ppmyryr)))

# Calculate the SR15 model values

# Generate the SR15 secular term
df_co2['SR15_secular'] = intercept
df_co2['SR15_secular'] += X['time1']*linear_coeff
# Include a factor of 0.5 in the quadratic coeff. (i.e., as in physics: (1/2)at^2 )
df_co2['SR15_secular'] += X['time2']*0.5*new_ppmyryr

# A version including the periodic model component
df_co2['SR15_ppm'] = df_co2['SR15_secular'] + df_co2['BAU_periodic']

# And create the "residual" of the SR15 and BAU model:
df_co2['SR15_resid'] = df_co2['SR15_ppm'] - df_co2['BAU_ppm']

# Create the SR15_periodic to be SR15_ppm minus BAU_secular:
# this is analogous to BAU_periodic which gives the BAU model minus the BAU secular term.
df_co2['SR15_periodic'] = df_co2['SR15_ppm'] - df_co2['BAU_secular']

# Output and check the SR15 model ppm and ppm/dt at some time in future
this_yr = 2022.20   # three years from GRETA

# Show all the df values at this_yr and the following week:
##print((df_co2[df_co2['time'] > this_yr]).head(2))

# Calculate the ppm/yr values for BAU and SR15 models:
one_week = (df_co2[df_co2['time'] > this_yr]).iloc[0]
next_week = (df_co2[df_co2['time'] > this_yr]).iloc[1]

bau_ppmyr = (next_week['BAU_secular']- one_week['BAU_secular'])/(next_week['time']- one_week['time'])
sr15_ppmyr = (next_week['SR15_secular']- one_week['SR15_secular'])/(next_week['time']- one_week['time'])

# Calculate the ppm/yr/yr values too
bau_ppmyryr = (bau_ppmyr - linear_coeff)/(this_yr - yr_fit)
sr15_ppmyryr = (sr15_ppmyr - linear_coeff)/(this_yr - yr_fit)

print("Check the models at t =",this_yr," :")
print("          ppm     ppm/yr    ppm/yr/yr")
print(" BAU:   {:.3f}    {:.3f}    +{:.3f}".format(one_week['BAU_ppm'], bau_ppmyr, bau_ppmyryr))
print("SR15:   {:.3f}    {:.3f}    {:.3f}".format(one_week['SR15_ppm'], sr15_ppmyr, sr15_ppmyryr))


# determine the time, etc of the newest measurement
yr_newest = max(df_co2.loc[df_co2['CO2_ppm'] > 0.0,'time'])
newest_row = (df_co2[df_co2['time'] > (yr_newest - 0.001)]).iloc[0]
# and one year before that
yr_ago_row = (df_co2[df_co2['time'] > (yr_newest - 1.001)]).iloc[0]

# get specific newest values
GTwk_newest = int(newest_row['GTwk'])
ppm_newest = newest_row['CO2_ppm']
bau_newest = newest_row['BAU_ppm']
sr15_newest = newest_row['SR15_ppm']
secular_newest = newest_row['CO2_secular']
resid_newest = newest_row['CO2_resid']
phase_newest = newest_row['yr_phase']
periodic_newest = newest_row['CO2_periodic']

print("\n#CO2 is "+
      "{:.2f} ppm [{:.2f} deseasoned] in GT week {} (t={:.2f}).".format(
          ppm_newest, secular_newest, GTwk_newest, yr_newest))
print("{:.2f} ppm from BAU ({:.2f}).".format(resid_newest,bau_newest))

# some yr_ago values:
bau_yr_ago = yr_ago_row['BAU_ppm']
yr_ago_time = yr_ago_row['time']
# 1 year increase, measured from BaU a year ago:
since_yr_ago = ppm_newest - bau_yr_ago
# This is not so relevant...
##print("+{:.2f} ppm from BAU of 1-year-ago ({:.2f}).  ".format(since_yr_ago, bau_yr_ago) )
print("Newest Rows:- ",newest_row)

# Colors for components
fit_data_clr = 'blue'
new_data_clr = 'green'
bau_model_clr = 'salmon'
sr15_model_clr = 'limegreen'
# used further below:
met_ppm_clr = 'darkorange'

# - - - - - 
# Overview Plot

# The models
ax = df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=3,figsize=(12,6))
# The SR15 secular trend
df_co2[~fit_rows].plot.scatter('time','SR15_secular',c=sr15_model_clr,s=3,ax=ax)
# The SR15 full model
df_co2[~fit_rows].plot.scatter('time','SR15_ppm',c=sr15_model_clr,s=3,ax=ax)
# BAU on top
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=3,ax=ax)
df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=3,ax=ax)

# data used for fitting
df_co2[fit_rows].plot.scatter('time','CO2_ppm',c=fit_data_clr,s=3,ax=ax)
# new data
df_co2[~fit_rows].plot.scatter('time','CO2_ppm',c=new_data_clr,s=6, yerr=resid_std, ax=ax)

# Plot limits
plt.xlim([yr_start,yr_end])
##plt.ylim([,])
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

plt.title("CO2 Weekly Data and Model.  Fit Data from {:.2f} to {:.2f} (".format(yr_start, yr_fit) + 
          fit_data_clr+")." +
         "  Recent Data ("+new_data_clr+")")
# GRETA
plt.plot([yr_GRETA,yr_GRETA],[407,419],c="gray")
plt.text(yr_GRETA+0.05, 420.0,"15 Mar 2019",color="gray")
plt.text(yr_GRETA+0.05, 418.2,"GRETA", color="gray")
# GT week 1
plt.plot([yr_GTwk1,yr_GTwk1],[407-4,419-3],c="gray")
plt.text(yr_GTwk1+0.05, 402.0,"GT week 1", color="gray")

# Show which data set this is from and the fit information
plt.text(0.06-0.03, 0.86, "Mauna Loa", color=fit_data_clr,
         transform=ax.transAxes, fontsize=18)
info_str = ("Fit: {:.3f}".format(linear_coeff) + 
            " & {:.3f}".format(ppmyryr_coeff) +
           ", sigma = {:.3f} ppm".format(resid_std))
plt.text(0.06+0.15, 0.86, info_str, color=fit_data_clr,
         transform=ax.transAxes, fontsize=14)

# Kaggle notebook url
plt.text(2016.0, 385.0, "s/wco2-after-greta", color="gray")

plt.ylabel("CO2 (ppm)")
plt.xlabel("Time")
#plt.savefig("4CO2afterGRETA_10yrs_"+co2_file_date+".png")
# plt.show()


# - - - - - 
# Number of valid-data-days in each Week

ax = df_co2.plot.scatter('time','days',c=fit_data_clr,s=3,figsize=(12,6))
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)
plt.title("Number of Valid Data Days in each week")
plt.ylabel("Number Valid")
plt.xlabel("Time")
#plt.savefig('5Sctter.png')
# plt.show()


# - - - - - 
# Trends Plot

# The Secular models
ax = df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,alpha=0.5,s=3, figsize=(12,6))
df_co2[~fit_rows].plot.scatter('time','SR15_secular',c=sr15_model_clr,s=3,ax=ax)
# BAU on top:
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=3,ax=ax)

df_co2[fit_rows].plot.scatter('time','CO2_secular',c=fit_data_clr,s=3,ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_secular',c=new_data_clr,s=6, ax=ax)
                               ##yerr=resid_std)   # include error bars on new data?

# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

plt.title("CO2 Weekly Deseasoned Data/Model.  Fit Data from {:.2f} to {:.2f} (".format(yr_start, yr_fit) +
         fit_data_clr+")  Recent Data ("+new_data_clr+")")

#plt.savefig("6CO2afterGRETA_AllDeseasoned_"+co2_file_date+".png", bbox_inches='tight')
# plt.show()

# - - - - - 
# Residuals Plot

# The residuals between data and model
# Model of the residuals = 0:
ax = df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,alpha=0.5,s=3, figsize=(12,6))
# Residuals for SR15 model:
df_co2[~fit_rows].plot.scatter('time','SR15_resid',c=sr15_model_clr,s=3,ax=ax)
# BAU on top:
df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,s=3,ax=ax)

# residuals of fit data
df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,alpha=0.15,s=1, yerr=resid_std, ax=ax)
df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,s=3,ax=ax)
# residuals of the new data
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,alpha=0.3,s=1, yerr=resid_std, ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,s=10, ax=ax)

# Plot limits
plt.xlim([yr_start,yr_end])
plt.ylim([-3.0,2.0])
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

# Show which data set this is
plt.text(0.06-0.03, 0.20, "Mauna Loa", color=fit_data_clr,
         transform=ax.transAxes, fontsize=18)
info_str = ("Fit: {:.3f}".format(linear_coeff) + 
            " & {:.3f}".format(ppmyryr_coeff) +
           ", sigma = {:.3f} ppm".format(resid_std))
plt.text(0.06+0.15, 0.20, info_str, color=fit_data_clr,
         transform=ax.transAxes, fontsize=14)

plt.title("CO2 Model Fit Residuals ("+fit_data_clr+", {:.2f}--{:.2f}) and Recent Data (".format(
        yr_start, yr_fit) + new_data_clr + ")")

#plt.savefig("7CO2afterGRETA_AllResids_"+co2_file_date+".png", bbox_inches='tight')
# plt.show()

# - - - - - 
# Periodic Plot

# The periodic model components
ax = df_co2.plot.scatter('yr_phase','BAU_periodic',c=bau_model_clr,alpha=0.5,s=3, figsize=(12,6))
df_co2.plot.scatter('yr_phase','BAU_periodic',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)
# The SR15 model minus the BAU_secular (only post-GRETA):
df_co2[~fit_rows].plot.scatter('yr_phase','SR15_periodic',c=sr15_model_clr,s=10,ax=ax)

# The fit data points (without secular term)
df_co2[fit_rows].plot.scatter('yr_phase','CO2_periodic',c=fit_data_clr,s=3,ax=ax)
# and the post-GRETA data points
df_co2[~fit_rows].plot.scatter('yr_phase','CO2_periodic',c=new_data_clr,s=6, yerr=resid_std, ax=ax)
df_co2[~fit_rows].plot.scatter('yr_phase','CO2_periodic',c=new_data_clr,s=20,ax=ax)

# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

plt.title("CO2 Weekly Periodic Data/Model.  Fit Data from {:.2f} to {:.2f} (".format(yr_start, yr_fit) +
          fit_data_clr+")." +
          "  Post-GRETA data ("+new_data_clr+")")

# plt.show()

# Routines to add some labels to the plots

# Data fit for prediction
def describe_fit_data(xfrac, yfrac, ax):

    """
    describe_fit_data(xfrac, yfrac, ax):

    Description:
    This function takes in three arguments - xfrac, yfrac and ax - and generates text labels to describe the pre-GRETA data used to fit the BAU Model. The labels are added to the plot specified by the input axis object.

    Args:
    xfrac (float): The x-coordinate of the text label as a fraction of the plot's width.
    yfrac (float): The y-coordinate of the text label as a fraction of the plot's height.
    ax (matplotlib.axes._subplots.AxesSubplot): The plot on which to add the text labels.

    Returns:
    None.
    """

    spacing = 0.04
    fontsize = 14
    plt.text(xfrac, yfrac, "Pre-GRETA Data ("+fit_data_clr+")",color=fit_data_clr,
         transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac, yfrac-spacing, "  Previous {} years".format(int(0.50+yr_fit - yr_start)),color=fit_data_clr,
         transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac, yfrac-2*spacing, "used to fit the BAU Model",color=fit_data_clr,
         transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac, yfrac-3*spacing, "(Source: NOAA ESRL)",color=fit_data_clr,
         transform=ax.transAxes, fontsize=fontsize)

# BAU model info
def describe_bau(xfrac, yfrac, ax):
    spacing = 0.04
    fontsize = 14
    plt.text(xfrac, yfrac, "Business as Usual Model ",
             color=bau_model_clr, transform=ax.transAxes, fontsize=fontsize)
    ##plt.text(xfrac+0.020, yfrac-spacing, "Linear:  {:.3f} ppm/yr".format(coeffs[0,0]),
    ##         color=bau_model_clr, transform=ax.transAxes, fontsize=fontsize)
    ##plt.text(xfrac-0.02, yfrac-2*spacing, "  dLinear/dt:  {:.3f} ppm/yr /yr".format(2.0*coeffs[0,1]),
    ##         color=bau_model_clr, transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac-0.01, yfrac-spacing, "Close to worst-case RCP8.5",
             color=bau_model_clr, transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac-0.01, yfrac-2*spacing, "500 ppm and rising in 2045",
             color=bau_model_clr, transform=ax.transAxes, fontsize=fontsize)
    
# IPCC SR15 model
def describe_sr15(xfrac, yfrac, ax):

    """
    describe_bau(xfrac, yfrac, ax):

    Description:
    This function takes in three arguments - xfrac, yfrac and ax - and generates text labels to describe the Business as Usual (BAU) Model. The labels are added to the plot specified by the input axis object.

    Args:
    xfrac (float): The x-coordinate of the text label as a fraction of the plot's width.
    yfrac (float): The y-coordinate of the text label as a fraction of the plot's height.
    ax (matplotlib.axes._subplots.AxesSubplot): The plot on which to add the text labels.

    Returns:
    None. """

    spacing = 0.04
    fontsize = 14
    plt.text(xfrac, yfrac,"IPCC 1.5 deg. Model (P2)",color=sr15_model_clr,
         transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac+0.02, yfrac-spacing, "$-$6.7 %/yr emissions", color=sr15_model_clr,
         transform=ax.transAxes, fontsize=fontsize)
    ##plt.text(xfrac+0.02, yfrac-2*spacing,"gives {:.3f} ppm/yr/yr".format(new_ppmyryr),color=sr15_model_clr,
    ##     transform=ax.transAxes, fontsize=fontsize)
    plt.text(xfrac+0.02, yfrac-2*spacing,"Maximum of 440 ppm",color=sr15_model_clr,
         transform=ax.transAxes, fontsize=fontsize)
    
# And the file date string:
co2_file_date_str = co2_file_date[:-7]+" "+co2_file_date[-7:-4]+" "+co2_file_date[-4:]

# - - - - -
# This plot focusses on the recent past and future year or so...
# Include labels for significant events.

# Starting and stopping years on plot
yr_plt_start = 2020.25 + 1.0
yr_plt_stop = 2022.75 + 1.0

# Model and Data, ppm values 
# show the BAU model and set the y-axis limits
ax = df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=3,figsize=(14,8),
                         xlim=(yr_plt_start,yr_plt_stop), ylim=(409.9+2, 424.1+2))
# with error bars
df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)

# Data used for fitting the model, w/errors - this pre-dates the range of this plot
##df_co2[fit_rows].plot.scatter('time','CO2_ppm',c=fit_data_clr,alpha=0.3,s=1,yerr=resid_std,ax=ax)
##df_co2[fit_rows].plot.scatter('time','CO2_ppm',c=fit_data_clr,s=12,ax=ax)

# Post-GRETA (aka new era) data, w/errors
df_co2[~fit_rows].plot.scatter('time','CO2_ppm',c=new_data_clr,alpha=0.5,yerr=resid_std,s=1, ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_ppm',c=new_data_clr,s=12, ax=ax)

# The IPCC SR15 full model
df_co2[~fit_rows].plot.scatter('time','SR15_ppm',c=sr15_model_clr,s=3,ax=ax)
# put BAU on top
df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=3,ax=ax)

# Add month names in (some parts of) some year(s)
for imon, month in enumerate(months):
    if imon >= 3:
        plt.text(2021.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    plt.text(2022.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    if imon <= 8:
        plt.text(2023.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

# title and axis labels
plt.title("CO$_2$ Weekly Data compared with a Fixed BAU Model" +
         " and an IPCC/SR1.5 Model -- Data through "+co2_file_date_str)

plt.ylabel("CO$_2$ (ppm, Mauna Loa, NOAA ESRL)")
plt.xlabel("Time")


# Add various annotations on the plot

# Label with the key information
info_string = ("CO$_2$ is  {:.2f} ppm  in the week ending  ".format(ppm_newest) +
          co2_file_date[:-7]+" "+co2_file_date[-7:-4]+" "+co2_file_date[-4:])
plt.text(2021.28, 425.2, info_string, color=new_data_clr, fontsize=15)
#
# State the relation to BAU:
# would like to have this:
abovebelow = ', *below*'
# use 'from' if only a little below:
if resid_newest > -0.30:
    abovebelow = 'from'
if resid_newest > 0:
    abovebelow = 'above'
rel_bau_string = ("$ \\longrightarrow $ {:.2f} ppm ".format(resid_newest) + 
                  abovebelow + " BAU.")
plt.text(2021.78, 424.4, rel_bau_string, color=new_data_clr, fontsize=15)


# Current GT week:   move this now and then
ppm_label = 419.25
yr_label = 2023.30
# line segment to it and text
plt.plot([yr_newest+0.010, yr_label+0.18],[ppm_newest-0.00, ppm_label+0.40],
         c="gray")
plt.text(yr_label+0.02, ppm_label-0.08,"Wk {}".format(GTwk_newest), fontsize=14, color='black')
# include the ppm residual-from-BAU value:
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(yr_label-0.04, ppm_label-0.65, "{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         fontsize=14, color='black')

# Dates/events of note:


# Friday 24 Sep 2021: Global Climate Strike  
# https://fridaysforfuture.org/September24/   #UprootTheSystem
yr_GRETA_UTS = 2021.0 + 267.0/DAYS_YR
plt.plot([yr_GRETA_UTS, yr_GRETA_UTS-0.10], [414.3,420.4],c="gray")
plt.text(yr_GRETA_UTS-0.30, 421.3,"Global Climate Strike, 24 Sep", color="gray")
plt.text(yr_GRETA_UTS-0.25, 421.3-0.5,"#UprootTheSystem", color="gray")

# Friday (29 Oct 2021, day 302) before COP26 (1 to 12 November 2021)
# Changed to End of COP26, day 316
yr_COP26 = 2021.0 + 316.0/DAYS_YR
plt.plot([yr_COP26-0.0,yr_COP26-0.03], [416.3,419.5-1.45],c="gray")
plt.text(yr_COP26-0.09, 419.5-0.7,"End of", color="gray")
plt.text(yr_COP26-0.09, 419.5-1.2,"COP26", color="gray") 

# Global Climate Strike #PeopleNotProfit Friday 25 Mar 2022, day 84)
# ~ Three years since first GRETA.
yr_GRETA_PNP = 2022.0 + 84.0/DAYS_YR
plt.plot([yr_GRETA_PNP-0.0,yr_GRETA_PNP-0.08], [422.2, 423.2-0.5],c="gray")
plt.plot([yr_GRETA_PNP-0.0,yr_GRETA_PNP-0.0],
         [419.0, 422.2],c="gray")
plt.text(yr_GRETA_PNP-0.49, 423.2,"Global Climate Strike", 
         color="gray", fontsize=14)
plt.text(yr_GRETA_PNP-0.49, 423.2-0.6,"#PeopleNotProfit", 
         color="gray", fontsize=14)

# Friday 23 Sep 2022: Global Climate Strike  
# https://fridaysforfuture.org/September23/   #PeopleNotProfit
yr_GRETA_PNP2 = 2022.0 + 266.0/DAYS_YR
plt.plot([yr_GRETA_PNP2, yr_GRETA_PNP2-0.0], [417.8,422.3],c="gray")
plt.plot([yr_GRETA_PNP2-0.0,yr_GRETA_PNP2-0.0], [417.7,414.5],
             c="gray",linestyle='dotted')
plt.text(yr_GRETA_PNP2-0.30, 423.3,"Global Climate Strike, 23 Sep",
             color="gray", fontsize=14)
plt.text(yr_GRETA_PNP2-0.210, 423.3-0.65,"#PeopleNotProfit",
             color="gray", fontsize=14)

# COP27 (7 to 18 November 2022)
# Friday before COP 27, Nov.4, day 308
# Friday end of COP 27, Nov.18, day 322
##yr_COP27 = 2022.0 + 322.0/DAYS_YR
##plt.text(yr_COP27-0.12+0.43, 419.4,"Friday end of", color="gray")
##plt.text(yr_COP27-0.12+0.43, 419.4-0.5,"COP27, 18 Nov", color="gray")
##plt.plot([yr_COP27-0.0,yr_COP27+0.28], [417.6,419.1],c="gray",linestyle='dotted')
##plt.plot([yr_COP27-0.0,yr_COP27-0.0], [418.3,416.5],c="gray",linestyle='dotted')

# Global Strike, 3 March 2023 -- 4 years since first global strike.
yr_GRETA_4YR = 2023.0 +62.0/DAYS_YR
plt.text(yr_GRETA_4YR-0.42, 425.2,"Global Strike, 3 Mar", color="gray", fontsize=14)
plt.text(yr_GRETA_4YR-0.42, 425.2-0.65,"#PeopleNotProfit", color="gray", fontsize=14)
plt.plot([yr_GRETA_4YR,yr_GRETA_4YR], [425.2-0.3,424.00],c="gray")
plt.plot([yr_GRETA_4YR,yr_GRETA_4YR], [420.75,422.75],c="gray",linestyle='dotted')


# Future events:


# Annotation text for the data, model, etc.
# BAU information
describe_bau(0.70, 0.39, ax)
# IPCC SR15 information
describe_sr15(0.70, 0.25, ax)    

# Kaggle notebook url
plt.text(2022.70, 412.8, "s/wco2-after-greta", color="gray")

#plt.savefig("8CO2afterGRETA_weeks_"+co2_file_date+".png", bbox_inches='tight')
plt.show()

# Simplified figure showing where we are compared to BAU and IPCC/SR15
# Plotted over the near past/future
# Strangely, * It seems too zoomed in to be useful? *
yr_plt_start = yr_newest - 1.5 - 0.2
yr_plt_stop = yr_newest + 0.7 - 0.2

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize':(12,6)})
fig = plt.figure(1,facecolor='white')
ax = fig.add_axes([0.1, 0.1, 0.8, 1.0])

# Show the BaU model line with errors size indicated
df_co2.plot('time','BAU_resid',c=bau_model_clr,legend=None,
                    xlim=(yr_plt_start, yr_plt_stop),ylim=(-3.65,1.0),ax=ax)
df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)

# The IPCC SR15 model line:
df_co2.plot('time','SR15_resid',c=sr15_model_clr,legend=None,
                    xlim=(yr_plt_start, yr_plt_stop),ax=ax)


# data used for fitting (not in this range)
##df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,alpha=0.3,yerr=resid_std,s=1,ax=ax)
##df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,s=12, ax=ax)

# new data
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,alpha=0.5,yerr=resid_std,s=1,ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,s=12, ax=ax)

# Add month names in the previous 2, current, next years
y_months = -1.95 -0.5 -0.5 -0.5
for year_offset in [-2,-1.0,0.0,1.0]:
    for imon, month in enumerate(months):
        x_month = int(yr_newest) + year_offset + 0.0417+imon/12.0
        if (x_month > (yr_plt_start + 0.0417)) and (x_month < (yr_plt_stop - 0.0417)):
            month_str = month+"\n'"+str(int(x_month-2000))
            plt.text(x_month, y_months, month_str, color=met_ppm_clr,
                horizontalalignment='center',fontsize=12)

# First GT week:
##plt.text(yr_GTwk1+0.01, -1.9,"Wk {}".format(1), color=fit_data_clr)
##plt.plot([yr_GTwk1, yr_GTwk1+0.015],[-0.4,-1.5],c="gray")

# Label the most recent value:
annot_resid = -1.2
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(yr_newest+0.40-0.2, annot_resid,"{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         color='black', fontsize=16)
plt.plot([yr_newest+0.025, yr_newest+0.40-0.015-0.2],
         [resid_newest,annot_resid+0.05],c="gray")

# Label with the key information
info_string = ("CO$_2$ is  {:.2f} ppm  in the week ending  ".format(ppm_newest) +
          co2_file_date_str+",")
plt.text(yr_newest - 1.25-0.2, 1.30 -0.5, info_string, color=new_data_clr, fontsize=18)

# State the relation to BAU:
# would like to have this:
abovebelow = ', *below*'
# use 'from' if only a little below:
if resid_newest > -0.30:
    abovebelow = 'from'
if resid_newest > 0:
    abovebelow = 'above'
rel_bau_string = ("$ \\longrightarrow $ {:.2f} ppm ".format(resid_newest) + 
                  abovebelow + " BAU.")
plt.text(yr_newest - 0.25-0.2, 1.08 - 0.5, rel_bau_string, color='black', fontsize=18)

# Include the GT week
plt.text( yr_newest+0.47-0.2, 1.32 - 0.5, "GT Wk {}".format(GTwk_newest),color='gray',fontsize=12)

# Current BAU ppm for reference
plt.text(yr_newest-1.40-0.2, 1.07 - 0.5, "This week's BAU is {:.2f} ppm".format(bau_newest),
         color=bau_model_clr, fontsize=16)

# Label the BAU and IPCC curves
plt.text(yr_newest+0.02, 0.10, "      BAU path:\n500 ppm in 2045",
         color=bau_model_clr, fontsize=16)
# Usually just manually adjust the annotation positions, but here for fun,
# adjust the y location based on where the IPCC curve leaves the plot:
sr15_at_end = df_co2['SR15_resid'].where(df_co2['time'] > yr_plt_stop).max()
# Nov, 2022: The IPCC curve is dropping and the data are not dropping with it  :(
plt.text(yr_newest+0.04, sr15_at_end+0.80, "IPCC 1.5 C path:\n~ 440 ppm max",
         color=sr15_model_clr, fontsize=16)


# Plot title
plt.title("Is the CO$_2$ moving below the reference BAU trajectory?", size=16)

# Plot grid and labels
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)
plt.xlabel("Data shown for the past 1.5 years", size=16)
ax.set_xticklabels([])
plt.ylabel("CO$_2 -$ BAU  (ppm)", size=16)


#plt.savefig("9CO2afterGRETA_simple_"+co2_file_date+".png", bbox_inches='tight')
plt.show()

# Repeat of the newest information
print("\nMost recent data point is ( "+
      "{:.2f}, {:.2f} ) for GT week {}.".format(yr_newest, ppm_newest, GTwk_newest))
print("{:.2f} ppm from current BAU ({:.2f}).".format(resid_newest,bau_newest))

# The fit routines
from scipy.optimize import leastsq, curve_fit

# The fit function
def fit_shift(x,t1,r1,t2,r2):

    """
        fit_shift(x, t1, r1, t2, r2)

        Description:
        This function takes in five arguments - x, t1, r1, t2, and r2 - and calculates a linearly-varying function that is r1 below t1 and r2 above t2. The output of the function is stored in the "out" array.

        Args:
        x (numpy.ndarray): An array of values to be used as the independent variable for the function.
        t1 (float): The first time point at which the function begins to increase.
        r1 (float): The initial value of the function.
        t2 (float): The second time point at which the function starts decreasing.
        r2 (float): The final value of the function.

        Returns:
        numpy.ndarray: An array of values representing the output of the linearly-varying function.
"""
    out = 0.0*x
    deltat = t2 - t1
    deltar = r2 - r1
    for tj,tval in enumerate(x):
        ##print(tj, tval)
        out[tj] = r1
        if tval > t1:
            if tval > t2:
                out[tj] = r2
            else:
                out[tj] = r1 + (deltar)*(tval - t1)/deltat
    return out

# Check the fit function
##fit_shift(np.array([10,12,14,16,18,20,22,24,26,28,30]),15,10,25,-10)

# Fit the shift function
# Use data from the first GRETA to the most recent value.
yr_plt_start = yr_GRETA
yr_plt_stop = yr_newest + 0.0001

# Select the values to fit
times = df_co2[(df_co2.time > yr_plt_start) & (df_co2.time < yr_plt_stop)].time.values
resids = df_co2[(df_co2.time > yr_plt_start) & (df_co2.time < yr_plt_stop)].CO2_resid.values

# Do the fit:
t1,r1,t2,r2 = curve_fit(fit_shift, times, resids,
                            bounds=([2019.50,-1.0,2020.0,-2.0],[2021.5,1.0,yr_newest,1.0]))[0]
print("Fit params:",t1,r1,t2,r2)

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize':(12,6)})
fig = plt.figure(1,facecolor='white')
ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])

df_co2.plot.scatter('time','CO2_resid',legend=None,
                    xlim=(yr_plt_start-0.05, yr_plt_stop+0.25),
                    ylim=(-3.05,1.0),c=new_data_clr,s=12, ax=ax)

plt.plot(times, fit_shift(times, t1,r1,t2,r2),linewidth=2)

# # Show the begin/end dates of the shift
plt.text(t1-0.15,-0.9,months[int(12*(t1-int(t1)))]+" {}".format(int(t1)), color="blue", size=14)
plt.text(t2-0.15, r2-0.5,months[int(12*(t2-int(t2)))]+" {}".format(int(t2)), color="blue", size=14)

# Show the before/after ppm fit levels
plt.text(2019.85,r1+0.15,"$-${:.2f} ppm".format(abs(r1)), color="blue", size=16)
plt.text(t2-0.2,r2+0.15,"$-${:.2f} ppm".format(abs(r2)), color="blue", size=16)

# Kaggle notebook url
plt.text(2019.4, -2.25, "co2-after-greta", color="gray")

# Plot title
plt.title("COVID Effect: Residuals from BAU, fit with a linear-shift (blue)" +
          "   Data through "+co2_file_date_str, size=16)
# Plot grid and labels
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)
plt.xlabel("Time", size=16)
plt.ylabel("CO$_2 -$ BAU  (ppm)", size=16)
#plt.savefig("10CO2afterGRETA_shift_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


# - - - - -
#
# DE-SEASONED version
#
# This plot focusses on the recent past and future year or so...
# Include labels for significant events.

# Starting and stopping years on plot
yr_plt_start = 2019.20
yr_plt_stop = 2024.20

# Model and Data, ppm values 
# show the BAU model and set the y-axis limits
ax = df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=3,figsize=(14,8),
                         xlim=(yr_plt_start,yr_plt_stop), ylim=(411.9, 421.1))
# with error bars
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)

# Post-GRETA (aka new era) data, w/errors
df_co2[~fit_rows].plot.scatter('time','CO2_secular',c=new_data_clr,alpha=0.5,yerr=resid_std,s=1, ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_secular',c=new_data_clr,s=12, ax=ax)

# The IPCC SR15 full model
df_co2[~fit_rows].plot.scatter('time','SR15_secular',c=sr15_model_clr,s=3,ax=ax)
# put BAU on top
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=3,ax=ax)

# Add month names in (some parts of) some year(s)
for imon, month in enumerate(months):
    if imon >= 3:
        plt.text(2020.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    plt.text(2021.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    if imon <= 8:
        plt.text(2022.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

# title and axis labels
plt.title("*Deseasoned* CO$_2$ Data compared w/ Fixed BAU Model" +
         " and an IPCC/SR1.5 Model -- through "+co2_file_date_str)

plt.ylabel("*Deseasoned* CO$_2$ (ppm, Mauna Loa, NOAA ESRL)")
plt.xlabel("Time")


# Label with the key information
info_string = ("Deseasoned ppm is  {:.2f} in week ending  ".format(secular_newest) +
          co2_file_date[:-7]+" "+co2_file_date[-7:-4]+" "+co2_file_date[-4:])
plt.text(2020.33, 420.2, info_string, color=new_data_clr, fontsize=15)
#
# State the relation to BAU:
# would like to have this:
abovebelow = ', *below*'
# use 'from' if only a little below:
if resid_newest > -0.30:
    abovebelow = 'from'
if resid_newest > 0:
    abovebelow = 'above'
rel_bau_string = ("$ \\longrightarrow $ {:.2f} ppm ".format(resid_newest) + 
                  abovebelow + " BAU.")
#
plt.text(2020.83, 419.6, rel_bau_string, color=new_data_clr, fontsize=15)


# Current GT week:   move this now and then
ppm_label = 418.00
yr_label = 2022.90
# line segment to it and text
plt.plot([yr_newest+0.03, yr_label-0.0],[secular_newest-0.0, ppm_label-0.05],c="gray")
plt.text(yr_label+0.02, ppm_label-0.08,"Wk {}".format(GTwk_newest), fontsize=14, color='black')
# include the ppm residual-from-BAU value:
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(yr_label-0.02, ppm_label-0.45, "{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         fontsize=14, color='black')


# Friday (29 Oct 2021, day 302) before COP26 (1 to 12 November 2021)
# Changed to End of COP, day 316
yr_COP26 = 2021.0 + 316.0/DAYS_YR
plt.plot([yr_COP26-0.0,yr_COP26-0.0], [415.0,415.8],c="gray")
plt.text(yr_COP26-0.06, 414.8-0.2,"End of", color="gray")
plt.text(yr_COP26-0.06, 414.8-0.6,"COP26", color="gray")



# Annotation text for the data, model, etc.
# BAU information
describe_bau(0.65, 0.93, ax)
# IPCC SR15 information
describe_sr15(0.74, 0.35, ax)    

# Kaggle notebook url
plt.text(2020.35, 412.8, "co2-after-greta", color="gray")

#plt.savefig("11CO2afterGRETA_weeks_deseason_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


# - - - - -
# Model and Data RESIDUALS

# start full 10 years pre-GRETA:
yr_plt_start = 2011.99
# end the plot at:
yr_plt_stop = 2024.01+0.2

# show the BAU model's residuals curve (=0) and error bars
ax = df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,s=3, figsize=(14,8),
                        xlim=(yr_plt_start,yr_plt_stop), ylim=(-3.65,2.5))
df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)

# Residual curve for SR15 model:
df_co2[~fit_rows].plot.scatter('time','SR15_resid',c=sr15_model_clr,s=3,ax=ax)
# Put BAU on top
df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,s=3,ax=ax)
                    
# data used for fitting, w errors
df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,alpha=0.3,yerr=resid_std,s=1,ax=ax)
df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,s=12, ax=ax)

# new data, w errors
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,alpha=0.3,yerr=resid_std,s=1,ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,s=16, ax=ax)

# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

# title
plt.title("CO$_2$ Weekly Residuals of the Fixed BAU Model ("+fit_data_clr+"). " +
         " Recent, New-Era Residuals ("+new_data_clr+") through "+co2_file_date_str)

plt.ylabel("Residual = CO$_2$ - BAU (ppm)")
plt.xlabel("Time (year)")

# Most recent data point
plt.plot([yr_newest+0.10, 2023.70], [resid_newest, -1.10], c='gray')
plt.text(2022.8+0.2, -0.70, "Wk {}".format(GTwk_newest),color='black',fontsize=14)
# include the ppm residual value from BaU:
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(2022.8, -0.95, "{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         color='black',fontsize=14)

# school strike week 1
plt.text(yr_GTwk1-0.7, 1.8,"#FFF Wk 1", color="gray")
plt.plot([yr_GTwk1-0.1, yr_GTwk1+0.008],[1.65,0.1],c="gray")

# GRETA-General (with Adults)
# 2019 and future minima in Sept:
plt.text(yr_GRETA_General+0.1, 1.56,"20th", color="gray")
for yr_sept in [0,1,2,3]:
    plt.plot([yr_GRETA_General+yr_sept,yr_GRETA_General+0.2+yr_sept],[0.25,1.05],c="gray")
    plt.text(yr_GRETA_General+0.1+yr_sept, 1.36,"Sept.", color="gray")
    plt.text(yr_GRETA_General+0.1+yr_sept, 1.16,str(2019+yr_sept), color="gray")

# Annotation text for the data, model, etc.
# Pre-GRETA data used for fitting BAU
describe_fit_data(0.040, 0.23, ax)
# BAU information
describe_bau(0.05, 0.96, ax)
plt.text(0.28, 0.96, "$ \\longrightarrow $ at 0 ppm level",
         color=bau_model_clr, transform=ax.transAxes, fontsize=14)
# IPCC SR15 information
describe_sr15(0.64, 0.13, ax) 

# Kaggle kernel
plt.text(yr_plt_start+0.30, -3.3, "co2-after-greta", color="gray")

#plt.savefig("12CO2afterGRETA_resids_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


# - - - - -
#
# DE-SEASONED version
#
# This plot focusses on the recent past and future year or so...
# Include labels for significant events.

# Starting and stopping years on plot
yr_plt_start = 2019.20
yr_plt_stop = 2024.20

# Model and Data, ppm values 
# show the BAU model and set the y-axis limits
ax = df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=3,figsize=(14,8),
                         xlim=(yr_plt_start,yr_plt_stop), ylim=(411.9, 421.1))
# with error bars
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)

# Post-GRETA (aka new era) data, w/errors
df_co2[~fit_rows].plot.scatter('time','CO2_secular',c=new_data_clr,alpha=0.5,yerr=resid_std,s=1, ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_secular',c=new_data_clr,s=12, ax=ax)

# The IPCC SR15 full model
df_co2[~fit_rows].plot.scatter('time','SR15_secular',c=sr15_model_clr,s=3,ax=ax)
# put BAU on top
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=3,ax=ax)

# Add month names in (some parts of) some year(s)
for imon, month in enumerate(months):
    if imon >= 3:
        plt.text(2020.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    plt.text(2021.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    if imon <= 8:
        plt.text(2022.00+0.0+imon/12.0, 412.2, month,color=met_ppm_clr)
    
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

# title and axis labels
plt.title("*Deseasoned* CO$_2$ Data compared w/ Fixed BAU Model" +
         " and an IPCC/SR1.5 Model -- through "+co2_file_date_str)

plt.ylabel("*Deseasoned* CO$_2$ (ppm, Mauna Loa, NOAA ESRL)")
plt.xlabel("Time")


# Label with the key information
info_string = ("Deseasoned ppm is  {:.2f} in week ending  ".format(secular_newest) +
          co2_file_date[:-7]+" "+co2_file_date[-7:-4]+" "+co2_file_date[-4:])
plt.text(2020.33, 420.2, info_string, color=new_data_clr, fontsize=15)
#
# State the relation to BAU:
# would like to have this:
abovebelow = ', *below*'
# use 'from' if only a little below:
if resid_newest > -0.30:
    abovebelow = 'from'
if resid_newest > 0:
    abovebelow = 'above'
rel_bau_string = ("$ \\longrightarrow $ {:.2f} ppm ".format(resid_newest) + 
                  abovebelow + " BAU.")
#
plt.text(2020.83, 419.6, rel_bau_string, color=new_data_clr, fontsize=15)


# Current GT week:   move this now and then
ppm_label = 418.00
yr_label = 2022.90
# line segment to it and text
plt.plot([yr_newest+0.03, yr_label-0.0],[secular_newest-0.0, ppm_label-0.05],c="gray")
plt.text(yr_label+0.02, ppm_label-0.08,"Wk {}".format(GTwk_newest), fontsize=14, color='black')
# include the ppm residual-from-BAU value:
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(yr_label-0.02, ppm_label-0.45, "{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         fontsize=14, color='black')


# Friday (29 Oct 2021, day 302) before COP26 (1 to 12 November 2021)
# Changed to End of COP, day 316
yr_COP26 = 2021.0 + 316.0/DAYS_YR
plt.plot([yr_COP26-0.0,yr_COP26-0.0], [415.0,415.8],c="gray")
plt.text(yr_COP26-0.06, 414.8-0.2,"End of", color="gray")
plt.text(yr_COP26-0.06, 414.8-0.6,"COP26", color="gray")



# Annotation text for the data, model, etc.
# BAU information
describe_bau(0.65, 0.93, ax)
# IPCC SR15 information
describe_sr15(0.74, 0.35, ax)    

# Kaggle notebook url
plt.text(2020.35, 412.8, "co2-after-greta", color="gray")

#plt.savefig("13CO2afterGRETA_weeks_deseason_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


#------------------------------------------------------------------------------
# - - - - -
# Model and Data RESIDUALS

# start full 10 years pre-GRETA:
yr_plt_start = 2011.99
# end the plot at:
yr_plt_stop = 2024.01+0.2

# show the BAU model's residuals curve (=0) and error bars
ax = df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,s=3, figsize=(14,8),
                        xlim=(yr_plt_start,yr_plt_stop), ylim=(-3.65,2.5))
df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,alpha=0.3,s=1,yerr=model_std,ax=ax)

# Residual curve for SR15 model:
df_co2[~fit_rows].plot.scatter('time','SR15_resid',c=sr15_model_clr,s=3,ax=ax)
# Put BAU on top
df_co2.plot.scatter('time','BAU_resid',c=bau_model_clr,s=3,ax=ax)
                    
# data used for fitting, w errors
df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,alpha=0.3,yerr=resid_std,s=1,ax=ax)
df_co2[fit_rows].plot.scatter('time','CO2_resid',c=fit_data_clr,s=12, ax=ax)

# new data, w errors
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,alpha=0.3,yerr=resid_std,s=1,ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_resid',c=new_data_clr,s=16, ax=ax)

# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

# title
plt.title("CO$_2$ Weekly Residuals of the Fixed BAU Model ("+fit_data_clr+"). " +
         " Recent, New-Era Residuals ("+new_data_clr+") through "+co2_file_date_str)

plt.ylabel("Residual = CO$_2$ - BAU (ppm)")
plt.xlabel("Time (year)")

# Most recent data point
plt.plot([yr_newest+0.10, 2023.70], [resid_newest, -1.10], c='gray')
plt.text(2022.8+0.2, -0.70, "Wk {}".format(GTwk_newest),color='black',fontsize=14)
# include the ppm residual value from BaU:
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(2022.8, -0.95, "{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         color='black',fontsize=14)

# school strike week 1
plt.text(yr_GTwk1-0.7, 1.8,"#FFF Wk 1", color="gray")
plt.plot([yr_GTwk1-0.1, yr_GTwk1+0.008],[1.65,0.1],c="gray")

# GRETA-General (with Adults)
# 2019 and future minima in Sept:
plt.text(yr_GRETA_General+0.1, 1.56,"20th", color="gray")
for yr_sept in [0,1,2,3]:
    plt.plot([yr_GRETA_General+yr_sept,yr_GRETA_General+0.2+yr_sept],[0.25,1.05],c="gray")
    plt.text(yr_GRETA_General+0.1+yr_sept, 1.36,"Sept.", color="gray")
    plt.text(yr_GRETA_General+0.1+yr_sept, 1.16,str(2019+yr_sept), color="gray")

# Annotation text for the data, model, etc.
# Pre-GRETA data used for fitting BAU
describe_fit_data(0.040, 0.23, ax)
# BAU information
describe_bau(0.05, 0.96, ax)
plt.text(0.28, 0.96, "$ \\longrightarrow $ at 0 ppm level",
         color=bau_model_clr, transform=ax.transAxes, fontsize=14)
# IPCC SR15 information
describe_sr15(0.64, 0.13, ax) 

# Kaggle kernel
plt.text(yr_plt_start+0.30, -3.3, "co2-after-greta", color="gray")

#plt.savefig("14CO2afterGRETA_resids_"+co2_file_date+".png", bbox_inches='tight')
plt.show()

# - - - - -
# Larger-range view of CO2 values
# 

# start 2 years before school strike week 1:
yr_plt_start = yr_GTwk1 - 3.5/DAYS_YR - 2.0
yr_plt_stop = yr_GRETA+5.0

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize':(14,6)})
fig = plt.figure(1,facecolor='white')
ax = fig.add_axes([0.1, 0.1, 0.8, 1.0])


df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=3,
                        xlim=(yr_plt_start,yr_plt_stop), ylim=(400.0, 425.0),ax=ax)
# The SR15 secular trend
df_co2[~fit_rows].plot.scatter('time','SR15_secular',c=sr15_model_clr,s=1,ax=ax, alpha=0.3)
# The SR15 full model
df_co2[~fit_rows].plot.scatter('time','SR15_ppm',c=sr15_model_clr,s=1,alpha=0.2,ax=ax,yerr=model_std)
df_co2[~fit_rows].plot.scatter('time','SR15_ppm',c=sr15_model_clr,s=3,ax=ax)
# BAU on top
df_co2.plot.scatter('time','BAU_secular',c=bau_model_clr,s=1,ax=ax, alpha=0.3)
df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=1,alpha=0.2,ax=ax,yerr=model_std)
df_co2.plot.scatter('time','BAU_ppm',c=bau_model_clr,s=3,ax=ax)

# data used for fitting
df_co2[fit_rows].plot.scatter('time','CO2_ppm',c=fit_data_clr,alpha=0.3,yerr=resid_std,s=1,ax=ax)
df_co2[fit_rows].plot.scatter('time','CO2_ppm',c=fit_data_clr,s=4,ax=ax)

# new data
df_co2[~fit_rows].plot.scatter('time','CO2_ppm',c=new_data_clr,s=1, yerr=resid_std,alpha=0.3, ax=ax)
df_co2[~fit_rows].plot.scatter('time','CO2_ppm',c=new_data_clr,s=6, ax=ax)

# Data fit for prediction
plt.text(2016.9, 413.7, "Pre-GRETA Data ("+fit_data_clr+")",color=fit_data_clr)
plt.text(2016.9, 413.7-0.9, "  Previous {} years".format(int(0.50+yr_fit - yr_start)),color=fit_data_clr)
plt.text(2016.9-0.1, 413.7-1.8, "used to fit the BAU Model",color=fit_data_clr)
plt.text(2016.9, 413.7-2.7, "(Source: NOAA ESRL)",color=fit_data_clr)

# GT week 1
plt.plot([yr_GTwk1-0.10,yr_GTwk1],[404,406],c="gray")
plt.text(yr_GTwk1-0.30, 403.0,"GT Wk 1", color="gray")

# The GRETA location
plt.plot([yr_GRETA,yr_GRETA-0.15],[412.8, 416.0-0.5],c="gray")
plt.text(yr_GRETA-1.8, 416.0+2.70,"First GRETA :",color="gray")
plt.text(yr_GRETA-2.3, 416.0+1.50,"Global Response to Extreme Temperature Anomalies",color="gray")
plt.text(yr_GRETA-2.1, 416.0+0.30,"15 Mar 2019, #FridaysForFuture, Wk 30", color="gray")

# current GT week:
plt.plot([yr_newest+0.025, 2023.4],[ppm_newest-0.2,413.0+0.9],c="gray")
plt.text(2023.0, 413.0, "Wk {}".format(GTwk_newest), color='black') #color=new_data_clr)
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(2023.2-0.3, 413-0.8,"{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         color='black') #color=new_data_clr)

# Friday 23 Sep 2022: Global Climate Strike  
# https://fridaysforfuture.org/September23/   #PeopleNotProfit
yr_GRETA_PNP2 = 2022.0 + 266.0/DAYS_YR
plt.plot([yr_GRETA_PNP2, yr_GRETA_PNP2-0.0], [414.5,408.1+1.0],c="gray")
plt.plot([yr_GRETA_PNP2-0.0,yr_GRETA_PNP2-0.0], [417.7,414.5],
             c="gray",linestyle='dotted')
plt.text(yr_GRETA_PNP2-0.30, 408.1,"Global Climate Strike",
             color="gray", fontsize=12)
plt.text(yr_GRETA_PNP2-0.30+0.19, 408.1-0.75,"23 Sep 2022",
             color="gray", fontsize=12)
plt.text(yr_GRETA_PNP2-0.30+0.09, 408.1-1.5,"#PeopleNotProfit",
             color="gray", fontsize=12)

# Annotation text for the data, model, etc.
# BAU information
describe_bau(0.20, 0.93, ax)
# IPCC SR15 information
describe_sr15(0.48, 0.35, ax) 

# Kaggle kernel
plt.text(2021.1, 400.7, "co2-after-greta", color="gray")


# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)
plt.ylabel("CO$_2$ (ppm)")
plt.xlabel("Time (year)")
##plt.title("CO$_2$ Weekly Data and Fixed Future Models." +
##         "  Post-GRETA Data ("+new_data_clr+") through "+co2_file_date)
plt.title("CO$_2$ Weekly Data ("+fit_data_clr+", "+new_data_clr+") and Fixed BAU Model ("+bau_model_clr+"). " +
         " Post-GRETA Data ("+new_data_clr+") through "+co2_file_date_str)


# Removed residuals plot


#plt.savefig("15CO2afterGRETA_future_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


# - - - - -
# Model and Data PERIODIC component
# show the model
ax = df_co2.plot.scatter('yr_phase','BAU_periodic',c=bau_model_clr,alpha=0.35,s=3, figsize=(14,8),
                        xlim=(0.0,1.0), ylim=(-7,5))
df_co2.plot.scatter('yr_phase','BAU_periodic',c=bau_model_clr,alpha=0.10,s=1,yerr=model_std,ax=ax)

# the SR15 model_periodic = SR15_ppm - BAU_secular
# just show it for one year:
#
yr_periodic = 2023.0
sr15_periodic_clr = "lime"
periodic_rows = (df_co2['time'] > yr_periodic) & (df_co2['time'] < 1.0+yr_periodic)
df_co2[periodic_rows].plot.scatter('yr_phase','SR15_periodic',c=sr15_periodic_clr,
                                   alpha=0.6,s=8,yerr=model_std,ax=ax)
# IPCC SR15 model
plt.text(0.43+0.00, -4.6+0.0,"* Year {} *".format(yr_periodic),color=sr15_periodic_clr)
plt.text(0.43-0.03, -4.6-0.5,"IPCC 1.5 deg. Model (P2)",color=sr15_periodic_clr)
plt.text(0.43-0.018, -4.6-0.9, "{:.2f}%/yr emissions".format(-6.7), color=sr15_periodic_clr)


# Data used for fitting
df_co2[fit_rows].plot.scatter('yr_phase','CO2_periodic',
                              c=fit_data_clr,alpha=0.4, s=8, ax=ax)

# New data
df_co2[~fit_rows].plot.scatter('yr_phase','CO2_periodic',
                               c=new_data_clr,alpha=1.0,yerr=resid_std,s=2,ax=ax)
df_co2[~fit_rows].plot.scatter('yr_phase','CO2_periodic',
                               c=new_data_clr,s=20, ax=ax)

# Current GT week:
plt.text(0.06, 3.50,"Wk {},".format(GTwk_newest), color='red')
plt.scatter([phase_newest],[periodic_newest],c="red",s=40)
# include the ppm residual value from BaU:
sign_str = "$+$"
if resid_newest < 0.0:
    sign_str = "$-$"
plt.text(0.06-0.015, 3.10, "{}{:.2f} ppm".format(sign_str,abs(resid_newest)),
         color='red')

# The (first) GRETA location
phase_GRETA = yr_GRETA - int(yr_GRETA)
plt.plot([phase_GRETA, phase_GRETA], [-2.6, 0.0], c="gray")
plt.text(0.03+0.06, -2.7,"First GRETA:",color="gray")
plt.text(0.03, -3.2,"    Global Response to Extreme Temperature Anomalies",color="gray")
plt.text(0.03+0.06, -3.7,"15 Mar 2019, #FridaysForFuture, Week 30", color="gray")

# Data fit for prediction
plt.text(0.66, 1.5, "Pre-GRETA Data ("+fit_data_clr+")",color=fit_data_clr)
plt.text(0.66, 1.5-0.4, "  Previous {} years".format(int(0.50+yr_fit - yr_start)),color=fit_data_clr)
plt.text(0.66, 1.5-0.8, "used to fit the Ba\AU Model",color=fit_data_clr)
plt.text(0.66, 1.5-1.2, "(Source: NOAA ESRL)",color=fit_data_clr)

# Data added since GRETA
plt.text(0.53, 3.4, "Recent data, post-GRETA ("+new_data_clr+")",color=new_data_clr)

# Kaggle kernel
plt.text(0.02, -6.3, "co2-after-greta", color="gray")

# Add month names
for imon, month in enumerate(months):
    plt.text(0.03+imon/12.0, -6.85, month,color='gray')

# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)
   
plt.title("CO2 Weekly Data ("+fit_data_clr+") and Fixed BAU Model ("+bau_model_clr+"). " +
         " Post-GRETA Data ("+new_data_clr+") through "+co2_file_date_str)

plt.ylabel("CO2 Data, Seasonal Component only (ppm)")
plt.xlabel("Phase within a Year")

#plt.savefig("16CO2afterGRETA_seasonal_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


# - - - - - 
# Plot the Residuals (from BAU-to-SR15) vs the Year Phase
# Shows some consistant deviations from the model and times of more/less variation,
# e.g., August through October is more stable.

# Since the post-GREATA data seems to follow SR15 (as of 24 Jan 2021), use the residuals from that:
# Add a column which is the CO2_ppm - SR15_ppm, same as calculating: CO2_resid - SR15_resid
# It's equal to CO2_resid pre GRETA:
df_co2['CO2_SRresid'] = df_co2['CO2_resid']
# and gets adjusted by SR15 after GRETA
df_co2.loc[~fit_rows,'CO2_SRresid'] = (df_co2.loc[~fit_rows,'CO2_resid'] - 
                                            df_co2.loc[~fit_rows,'SR15_resid'])

# The newest week's residual from SR15:
SRresid_newest = ppm_newest - sr15_newest

# A zero line of points
ax = df_co2.plot.scatter('yr_phase','BAU_resid',c=bau_model_clr,alpha=0.5,s=3,
                         xlim=(-0.001,1.001), ylim=(-2.01,2.01), figsize=(14,8))

# The residuals of the fit data points for some previous years
prev_years = 2.0
df_co2[(df_co2.time < yr_fit) & (df_co2.time > (yr_fit - prev_years))].plot.scatter('yr_phase','CO2_resid',
                                c=fit_data_clr,s=20,yerr=resid_std,alpha=0.4,ax=ax)

# and the post-GRETA data points with nominal error bars
df_co2[~fit_rows].plot.scatter('yr_phase','CO2_SRresid',
                        c=new_data_clr,s=6, yerr=resid_std,alpha=0.8,ax=ax)
df_co2[~fit_rows].plot.scatter('yr_phase','CO2_SRresid',
                        c=new_data_clr,s=20,alpha=1.0,ax=ax)

# current GT week:
plt.text(phase_newest, 1.75, "Wk {}".format(GTwk_newest), color='red')
plt.scatter([phase_newest],[SRresid_newest],c="red",s=50)

# Add month names
for imon, month in enumerate(months):
    plt.text(0.03+imon/12.0, -1.9, month,color=met_ppm_clr)
    
# show grid(s)?
plt.grid(axis='y', color='gray', linestyle=':', linewidth=1)

plt.title("Residuals from SR15 model.  Pre-GRETA," +
          " {:.2f} to {:.2f} (".format(yr_fit - prev_years, yr_fit) +
          fit_data_clr+")." +
          "  Post-GRETA data ("+new_data_clr+") through "+co2_file_date_str)

#plt.savefig("17CO2afterGRETA_YearResids_"+co2_file_date+".png", bbox_inches='tight')
plt.show()


