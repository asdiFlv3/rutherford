#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
pi=np.pi
colours=['#FFBE7A', '#8ECFC9', '#FA7F6F', '#82B0D2']#colour design
#%%'''bias calculation'''
bias=np.array([0,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.5,3,3.5,4,4.5,5,5,5,6])
counts=np.array([30444,16953.2,48623, 75748, 73379, 100407, 54565, 1343, 767,758,747,747,804, 746, 691,753,741,736.4, 702.5,719.1,672.6,519.2,319.7,
    147.6,49.2])
counts_err=1/np.sqrt(counts)
countRate=counts/10
countRate_err=counts_err/10
countRate_filter=np.array([count for count in countRate if count<1000])
bias_filter=np.array([b for b in bias if b>1.1])

plt.figure()
plt.grid(True, which="major", linestyle="--", linewidth=0.8, color="#E7DAD2")
# Minor ticks
plt.grid(True, which="minor", linestyle="--", linewidth=0.5, color="#999999")
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1)) # Adjust interval for x minor grid
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(10)) # Adjust interval for y minor grid
plt.plot(bias_filter, countRate_filter)
plt.show()

#%% functions to test fitting

def chi_sq(par, x, y, yfit, err):
    resid= yfit - y
    residual_n = np.array([resid[i] / err[i] for i in range(len(resid))]) # residuals scaled by the error
    resum=np.sum((residual_n)**2) # sum of the squares of reduced residuals
    k = len(y) - len(par) # number of degrees of freedom
    redchi2 = resum/k # reduced chi^2
    return residual_n, k, redchi2


#%%angular dependence
angularError=0.5
data = np.genfromtxt('r2_data.csv', delimiter=',', skip_header=1, usecols=(0,1,2,3))
#extract data from each column
scatterAngle_deg = data[:, 0] #in degrees
timePeriod = data[:, 1]
count = data[:, 2]
countRate = data[:, 3]
#error calculation
countError = np.sqrt(count)
countRateError = countError/timePeriod
scatterAngle = scatterAngle_deg*pi/180
scatterAngleError = angularError*pi/180

#%%plot&fit raw data to find zero error

plt.errorbar(scatterAngle, countRate, yerr=countRateError, xerr=scatterAngleError, fmt=',', ecolor=colours[3], capsize=3, label = "experiment")
plt.legend()
plt.grid(True)
plt.xlabel(r"Scatter Angle(rad)")
plt.ylabel("Count Rate(/s)")
plt.show()


#%%log them and find power law
zeroError=-0.28*pi/180
scatterAngle_pos= scatterAngle[-18:] #save only the positive part for ln
x=np.log(1 / (np.sin( (scatterAngle_pos-zeroError)/2 ) ))
countRate_pos=countRate[-18:]
y=np.log(countRate_pos)
yerror=countRateError/countRate
yerr_pos=yerror[-18:]

print(x, y)

#fitting
#find datasets x<3.5

'''
to find power law i filtered the datasets with x>3.5 where it stops to show linear behaviour in fitting
however it still need to be take into consideration
'''

indices=np.where(x<3.5)[0]
x_filter=x[indices]
y_filter=y[indices]
yerr_filtered=yerr_pos[indices]

x_fit = np.linspace(0.9*min(x),3.5,100)
p, cvm = np.polyfit(x_filter, y_filter, 1, cov=True) #fit the data

fitline = np.poly1d(p)
y_fitted = np.polyval(fitline, x_filter)

residuals, dof, reduced_chisq=chi_sq(p, x_filter, y_filter, y_fitted, yerr_pos) # 0-scaled residual, 1-dof, 2-reduced chi-square 
coefficient_err = np.sqrt(np.diag(cvm))



print("slope is ", p, " with error ", coefficient_err)
print("Number of degrees of freedom is", dof)
print("Reduced chi^2 is", reduced_chisq)


#%% plot
plt.figure(dpi=300)
plt.errorbar(x, y, yerr=yerr_pos, fmt=',', ecolor=colours[3], capsize=3, label = 'experiment')
plt.plot(x_fit,fitline(x_fit),"-",label="Fit to polynomial", color = colours[2]) #plot fit line
plt.xlabel("ln(Scatter Angle(rad))")
plt.ylabel("ln[csc(Count Rate</s>)]")
plt.legend()
plt.grid(True)
plt.show()
