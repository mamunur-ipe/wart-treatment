"""
@author: Mamunur Rahman
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
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

        # scale the features        
        scaler  = StandardScaler().fit(self.X)
        X_scaled = scaler.transform(self.X)
        
        # keep the top 'n' features
        scores = calculate_F_score(self.X, self.y)
        idx = np.argsort(-np.array(scores))
        X=X_scaled[:, idx[0 : no_of_top_features]]
        
        
        #k-fold CV with oversampling
        np.random.seed(8)  #fix random seed number for reproducibility
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
                    
                        
                    # Over sampling
                    sm = ADASYN(random_state=111)
        #             sm = SMOTE(random_state=111)
        #             sm = BorderlineSMOTE(random_state=111)
        #             sm = KMeansSMOTE(random_state=1,cluster_balance_threshold=.3)
            
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                  
                    #train classifier
                    clf = svm.SVC(gamma= math.pow(2,i), C=math.pow(2,j), cache_size=2000, kernel='rbf')
        #             clf = svm.SVC(gamma= math.pow(2,i), C=math.pow(2,j), cache_size=2000, kernel='poly', degree=3)
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

    # perform Over sampling using different methods and create 3D plots
    def do_oversampling_and_plot(self):
    
        sm = SMOTE(random_state=111)
        ad = ADASYN(random_state=111)
        bs = BorderlineSMOTE(random_state=111)
        X_new_sm, y_new_sm = sm.fit_resample(X, y)
        X_new_ad, y_new_ad = ad.fit_resample(X, y)
        X_new_bs, y_new_bs = bs.fit_resample(X, y)
        
        
        #before oversampling
        data_old = np.concatenate((X, y.reshape(-1,1)), axis=1)
        
        data_1_old = data_old[data_old[:, 7] == 1] #data with class '1'
        data_0_old = data_old[data_old[:, 7] == 0] #data with class '0'
        
        #after oversampling (SMOTE)
        a = X_new_sm[:, 0:7] #first two columns
        b = y_new_sm.reshape(-1,1)  #class/target column
        data = np.concatenate((a, b), axis=1)
        
        data_1 = data[data[:, 7] == 1] #data with class '1'
        data_0 = data[data[:, 7] == 0] #data with class '0'
        
        #after oversampling (ADASYN)
        a_ad = X_new_ad[:, 0:7] #first two columns
        b_ad = y_new_ad.reshape(-1,1)  #class/target column
        data_ad = np.concatenate((a_ad, b_ad), axis=1)
        
        data_1_ad = data_ad[data_ad[:, 7] == 1] #data with class '1'
        data_0_ad = data_ad[data_ad[:, 7] == 0] #data with class '0'
        
        #after oversampling (Borderline_SMOTE)
        a_bs = X_new_bs[:, 0:7] #first two columns
        b_bs = y_new_bs.reshape(-1,1)  #class/target column
        data_bs = np.concatenate((a_bs, b_bs), axis=1)
        
        data_1_bs = data_bs[data_bs[:, 7] == 1] #data with class '1'
        data_0_bs = data_bs[data_bs[:, 7] == 0] #data with class '0'
        
        
        ### create 3D plot        
        fig = plt.figure(constrained_layout=True, figsize=(12,7.5))
        
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3 = fig.add_subplot(gs[1, 1], projection='3d')
        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        
        size = 10.5  #label font size
        
        ## scatter plot before oversampling
        scatter1=ax1.scatter(data_1_old[:, [1]], data_1_old[:, [2]], data_1_old[:, [4]], c='yellow', marker='o', s=40, edgecolors='k', depthshade= 0)
        scatter2=ax1.scatter(data_0_old[:, [1]], data_0_old[:, [2]], data_0_old[:, [4]], c='r', marker='o', s=40, edgecolors='k', depthshade= 0)
        
        ax1.set_xlabel('Age (years)', fontsize=size)
        ax1.set_ylabel('Elapsed time (months)', fontsize=size)
        ax1.set_zlabel('Wart type', fontsize=size)
        ax1.set_zticks(range(1,4,1))
        ax1.set_title('(a) Before oversampling\n', fontsize=14, fontweight='bold')
        # set legend
        legend=ax1.legend([scatter1, scatter2], ['Y=1 (Yes)', 'Y=0 (No)'], numpoints = 1, loc='best', fontsize=size)
        legend.get_frame().set_edgecolor('k')
        
        
        ## scatter plot after oversampling (SMOTE)
        scatter1=ax2.scatter(data_1[:, [1]], data_1[:, [2]], data_1[:, [4]], c='yellow', marker='o', s=40, edgecolors='k', depthshade= 0)
        scatter2=ax2.scatter(data_0[:, [1]], data_0[:, [2]], data_0[:, [4]], c='r', marker='o', s=40, edgecolors='k', depthshade= 0)
        
        ax2.set_xlabel('Age (years)', fontsize=size)
        ax2.set_ylabel('Elapsed time (months)', fontsize=size)
        ax2.set_zlabel('Wart type', fontsize=size)
        ax2.set_zticks(range(1,4,1))
        ax2.set_title('(b) After oversampling (SMOTE)\n', fontsize=14, fontweight='bold')
        # set legend
        legend=ax2.legend([scatter1, scatter2], ['Y=1 (Yes)', 'Y=0 (No)'], numpoints = 1, loc='best', fontsize=size)
        legend.get_frame().set_edgecolor('k')
        
        
        ## scatter plot after oversampling (ADASYN)
        scatter1=ax3.scatter(data_1_ad[:, [1]], data_1_ad[:, [2]], data_1_ad[:, [4]], c='yellow', marker='o', s=40, edgecolors='k', depthshade= 0)
        scatter2=ax3.scatter(data_0_ad[:, [1]], data_0_ad[:, [2]], data_0_ad[:, [4]], c='r', marker='o', s=40, edgecolors='k', depthshade= 0)
        
        ax3.set_xlabel('Age (years)', fontsize=size)
        ax3.set_ylabel('Elapsed time (months)', fontsize=size)
        ax3.set_zlabel('Wart type', fontsize=size)
        ax3.set_zticks(range(1,4,1))
        ax3.set_title('\n(d) After oversampling (ADASYN)\n', fontsize=14, fontweight='bold')
        # set legend
        legend=ax3.legend([scatter1, scatter2], ['Y=1 (Yes)', 'Y=0 (No)'], numpoints = 1, loc='best', fontsize=size)
        legend.get_frame().set_edgecolor('k')
        
        
        ## scatter plot after oversampling (ADASYN)
        scatter1=ax4.scatter(data_1_bs[:, [1]], data_1_bs[:, [2]], data_1_bs[:, [4]], c='yellow', marker='o', s=40, edgecolors='k', depthshade= 0)
        scatter2=ax4.scatter(data_0_bs[:, [1]], data_0_bs[:, [2]], data_0_bs[:, [4]], c='r', marker='o', s=40, edgecolors='k', depthshade= 0)
        
        ax4.set_xlabel('Age (years)', fontsize=size)
        ax4.set_ylabel('Elapsed time (months)', fontsize=size)
        ax4.set_zlabel('Wart type', fontsize=size)
        ax4.set_zticks(range(1,4,1))
        ax4.set_title('\n(c) After oversampling (Borderline-SMOTE)\n', fontsize=14, fontweight='bold')
        # set legend
        legend=ax4.legend([scatter1, scatter2], ['Y=1 (Yes)', 'Y=0 (No)'], numpoints = 1, loc='best', fontsize=size)
        legend.get_frame().set_edgecolor('k')
        
        
#        #save figure
#        fig.savefig("Oversampling plot_3D.png", dpi=300, bbox_inches='tight')
        plt.show()

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
        yes = [71, 0]  #class 1
        no = [0, 19]  #class 2
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
        yes = [39, 32]  #class 1
        no = [10, 9]  #class 2
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
        yes = [19, 23, 11, 18]  #class 1
        no = [2, 3, 7, 7]  #class 2
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
        yes = [11, 18, 29, 13]  #class 1
        no = [2, 2, 2, 13]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Time elapsed (months)', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('0≤time<3','3≤time<6','6≤time<9','9≤time'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 31, 10))
        
        
        ## Number of warts
        ax = fig.add_subplot(gs[1, 0])
        N = 4  #number of bars
        yes = [28, 16, 11, 16]  #class 1
        no = [4, 5, 7, 3]  #class 2
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
        yes = [36, 17, 18]  #class 1
        no = [11, 5, 3]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Wart types', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('Common','Plantar','Both'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 51, 10))
        
        
        ## Largest wart (sq. mm)
        ax = fig.add_subplot(gs[1, 2])
        N = 4  #number of bars
        yes = [33, 26, 2, 10]  #class 1
        no = [8, 8, 0, 3]  #class 2
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
        
        
        ## Induration diameter
        ax = fig.add_subplot(gs[1, 3])
        N = 4  #number of bars
        yes = [55, 4, 3, 9]  #class 1
        no = [14, 2, 0, 3]  #class 2
        ind = np.arange(N)    # the x locations for the groups
        p1 = ax.bar(ind, yes, width, color= 'palegreen', edgecolor='k')
        p2 = ax.bar(ind, no, width, bottom=yes, color= 'r', edgecolor='k')
        
        ax.set_xlabel('Induration diameter (mm)', fontsize=size1, color='b')
        ax.set_xticks(ind)
        ax.set_xticklabels(('0≤dia<20','20≤dia<30','30≤dia<40','40≤dia'), fontsize=size2)
        for label in ax.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        ax.set_yticks(np.arange(0, 71, 10))
        
        
        ##super title of the subplots
        fig.suptitle("(a) Immunotherapy", fontweight="bold", size=12)
        plt.show()
        
        #fig.savefig("feature plot_immunotherapy.png", dpi=300, bbox_inches='tight')

#-------------------------------------------------------------------------------------------------------   

    ## create bar chart for the feature ranking
    def plot_feature_ranking(self):  
   
        scores = calculate_F_score(X, y)
        column_names = list(pd.read_csv('Data/Immunotherapy.csv').columns)[0:-1] #dropped the target column
        
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
        ax.set_title('(a) Immunotherapy', fontweight="bold", fontsize=size)
        
        plt.show()
    #    #save figure
    #    fig.savefig("Feature score_immunotherapy.png", dpi=300, bbox_inches='tight')        

#------------------------------------------------------------------------------------------------------- 
             
        
        
if __name__ == '__main__':

    df=pd.read_csv('Immunotherapy.csv')
    X=df.iloc[:, 0:7].values
    y=df.iloc[:,7].values

    clf = Classifier(X, y)
    
    # plot descriptive stats
    clf.descriptive_analysis_plot() 

    ## create bar chart for the feature ranking
    clf.plot_feature_ranking()
    
    clf.do_oversampling_and_plot()
    
    clf.build_model(no_of_top_features = 3)
    




