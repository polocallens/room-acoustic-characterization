import pandas as pd
import matplotlib.pyplot as plt


def plot_distrib_filtered(feature, feature_name, quantile = 0.98):
    feature = pd.DataFrame(feature)

    #test.plot.hist(bins = 100, alpha = 0.7,figsize=(15,15))
    high_lim = feature.quantile(quantile)
    data_filtered = feature[(feature < high_lim)]
    data_filtered.plot.hist(bins = 50, alpha = 0.7,figsize=(15,15), title = feature_name)
    plt.show()
    
    
def plot_scatter(feature_1,feature_2,feature_1_name = None, feature_2_name = None, title = None):
    feature_1 = np.array(feature_1)
    feature_1 = np.mean(feature_1,axis=1)
    
    plt.figure(figsize=(15,10))
    plt.xlabel(feature_1_name)
    plt.ylabel(feature_2_name)
    plt.suptitle(title)
    plt.xlim(left=-1,right=2)
    plt.ylim(bottom=-10,top=40)
    plt.scatter(feature_1,feature_2)    
    plt.show()
    
