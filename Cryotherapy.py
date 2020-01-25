"""
@author: Mamunur Rahman
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
import math
from F_score import calculate_F_score


class Classifier:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    ### build ML model and measure the performance
    def build_model(self, no_of_top_features):
        
        X= self.X
        y = self.y

        # scale the features        
        scaler  = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        
        # keep the top 'n' features
        scores = calculate_F_score(X, y)
        idx = np.argsort(-np.array(scores))
        X=X_scaled[:, idx[0 : no_of_top_features]]
        
        
        #k-fold CV with oversampling
        np.random.seed(42)  #fix random seed number for reproducibility
        # k-fold CV
        k_fold = StratifiedKFold(n_splits=10, shuffle=True) 
        
        # define an empty dataframe to store the perfromance measures
        performance = pd.DataFrame(columns = ['gamma', 'C', 'Accuracy', 'Sensitivity', 'Specificity'])
        
        counter = 0
        for i in np.linspace(-3, 4, 8):
            for j in np.linspace(-3, 4, 8):
                
                accu=[] #allocate memory for accuracy
                sen=[]  #allocate memory for sensitivity
                spe=[]  #allocate memory for specificity
                
                for train_index, test_index in k_fold.split(X, y):
                    X_train, y_train = X[train_index], y[train_index]
                    X_test, y_test = X[test_index], y[test_index]
                        
#                    # Over sampling
#                    sm = ADASYN(random_state=111)
#                     sm = SMOTE(random_state=111)
#                     sm = BorderlineSMOTE(random_state=111)
#                     sm = KMeansSMOTE(random_state=1,cluster_balance_threshold=.3)
#            
#                    X_train, y_train = sm.fit_resample(X_train, y_train)
                  
                    #train classifier
                    clf = svm.SVC(gamma= math.pow(2,i), C=math.pow(2,j), cache_size=2000, kernel='rbf')
#                    clf = svm.SVC(gamma= math.pow(2,i), C=math.pow(2,j), cache_size=2000, kernel='poly', degree=3)
                    clf = clf.fit(X_train, y_train)
        
                    #predict
                    y_pred=clf.predict(X_test)
                
                    #append accuracy_score to accuracy
                    accu.append(accuracy_score(y_test, y_pred))
                    #confusion matrix
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    sen.append(tp/(tp+fn))
                    spe.append(tn/(tn+fp))
                    
                # store data in 'performance' data frame
                performance.loc[counter, 'gamma'] = math.pow(2, i)
                performance.loc[counter, 'C'] = math.pow(2, j)
                performance.loc[counter, 'Accuracy'] = np.mean(accu)
                performance.loc[counter, 'Sensitivity'] = np.mean(sen)
                performance.loc[counter, 'Specificity'] = np.mean(spe)
                
                counter = counter + 1
                    
        # display the row with maximum Accuracy
        print(performance[performance.Accuracy == performance.Accuracy.max()])

#-------------------------------------------------------------------------------------------------------   
    ## create bar chart for the feature ranking
    def plot_feature_ranking(self):

        X= self.X
        y = self.y
   
        scores = calculate_F_score(X, y)
        column_names = list(pd.read_csv('Data/Cryotherapy.csv').columns)[0:-1] #dropped the target column
        
        ## create bar chart for the feature ranking
        fig = plt.figure(constrained_layout=True, figsize=(5,3.5))
        
        gs = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        
        size = 12  #label font size
        
        y_pos = np.arange(len(column_names))
        
        ax.barh(y_pos, scores, align='center', height=0.5, color= 'r', edgecolor='w')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(column_names, fontsize=size)
        ax.set_xlabel('F-score', fontsize=size)
        ax.set_title('(a) Cryotherapy', fontweight="bold", fontsize=size)
        
        plt.show()
    #    #save figure
    #    fig.savefig("Feature score_immunotherapy.png", dpi=300, bbox_inches='tight') 

#-------------------------------------------------------------------------------------------------
    # descriptive stats
    def descriptive_analysis_plot(self):
                    
        fig = plt.figure(constrained_layout= True, figsize=(10, 5.5))
        
        gs = GridSpec(2, 4, figure=fig)
        size1 = 11  #axis labels font size
        size2 = 10  #tick-labels font size
        width = 0.5       # the width of the bars
        
        ## Response variable
        ax = fig.add_subplot(gs[0, 0])
        N = 2  #number of bars
        yes = [48, 0]  #class 1
        no = [0, 42]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Treatment success', fontsize=size1, color='b')
        ax.set_ylabel('Number of cases', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('Y=1', 'Y=0'), fontsize=size2)
        ax.set_yticks(np.arange(0, 101, 20))
        # set legend
        legend=ax.legend([p1, p2], ['Y=1 (Yes)', 'Y=0 (No)'], loc='upper right', fontsize=10)
        legend.get_frame().set_edgecolor('k')
        
        
        ## Gender
        ax = fig.add_subplot(gs[0, 1])
        N = 2  #number of bars
        yes = [21, 27]  #class 1
        no = [22, 20]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Gender', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('Female', 'Male'), fontsize=size2)
        ax.set_yticks(np.arange(0, 51, 10))
        
        
        ## Age
        ax = fig.add_subplot(gs[0, 2])
        N = 4  #number of bars
        yes = [24, 16, 5, 3]  #class 1
        no = [3, 11, 16, 12]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Age (years)', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('10≤age<20','20≤age<30','30≤age<40','40≤age'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 31, 10))
        
        
        ## Time elapsed
        ax = fig.add_subplot(gs[0, 3])
        N = 4  #number of bars
        yes = [8, 22, 11, 7]  #class 1
        no = [1, 2, 6, 33]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Time elapsed (months)', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('0≤time<3','3≤time<6','6≤time<9','9≤time'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 41, 10))
        
        
        ## Number of warts
        ax = fig.add_subplot(gs[1, 0])
        N = 4  #number of bars
        yes = [21, 7, 9, 11]  #class 1
        no = [16, 13, 7, 6]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Number of warts', fontsize=size1, color='b')
        ax.set_ylabel('Number of cases', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('1-3','4-6','7-9','≥ 10'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 41, 10))
        
        
        ## Wart type
        ax = fig.add_subplot(gs[1, 1])
        N = 3  #number of bars
        yes = [38, 6, 4]  #class 1
        no = [16, 3, 23]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Wart types', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('Common','Plantar','Both'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 61, 10))
        
        
        ## Largest wart (sq. mm)
        ax = fig.add_subplot(gs[1, 2])
        N = 4  #number of bars
        yes = [17, 20, 8, 3]  #class 1
        no = [19, 10, 7, 6]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Largest wart area (sq. mm)', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('0≤area<50','50≤area<100','100≤area<150','150≤area'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 41, 10))
        
        
        ##super title of the subplots
        fig.suptitle("(b) Cryotherapy", fontweight="bold", size=12)
        plt.show()
        
#        fig.savefig("feature plot_cryotherapy.png", dpi=300, bbox_inches='tight')

#------------------------------------------------------------------------------------------------------- 
                 
        
if __name__ == '__main__':

    df=pd.read_csv('Data/Cryotherapy.csv')
    X=df.iloc[:, 0:6].values
    y=df.iloc[:,6].values

    clf = Classifier(X, y)
    
    # plot descriptive stats
    clf.descriptive_analysis_plot() 

    ## create bar chart for the feature ranking
    clf.plot_feature_ranking()
    
    clf.build_model(no_of_top_features = 4)
    




