# Wart-treatment decision support tool
Warts are noncancerous benign tumors caused by the Human Papilloma Virus. The success rates of cryotherapy and immunotherapy, two common treatment methods for cutaneous warts, are 44% and 72%, respectively. The treatment methods, therefore, fail to cure a significant percentage of the patients.

We aim to develop a reliable machine learning model to accurately predict the success of immunotherapy and cryotherapy for individual patients based on their demographic and clinical characteristics. We employed support vector machine (SVM) classifier utilizing a dataset of 180 patients who were suffering from various types of warts and received treatment either by immunotherapy or cryotherapy. We utilized three different oversampling methods to balance the minority class. F-score along with sequential backward selection (SBS) algorithm were utilized to extract the best set of features.

Our developed models provide better classification accuracy, sensitivity, and specificity compared to the models available in the literature. The developed methodology could potentially assist the dermatologists as a decision support tool by predicting the success of every unique patient before starting the treatment process.

To use the application, please visit the below link:
http://mamunur.pythonanywhere.com/input_wart
