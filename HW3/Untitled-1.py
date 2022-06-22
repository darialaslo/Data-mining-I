
import numpy as np

def confusion_matrix(y_true, y_pred):
    '''
    Function for calculating TP, FP, TN, and FN.
    The input includes the vector of true labels
    and the vector of predicted labels
    '''
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(len(y_true)):
        if (y_true[i]==1):
            if(y_true[i]==y_pred[i]):
                tp+=1
            else:
                fn+=1
        
        else:
            if(y_true[i]==y_pred[i]):
                tn+=1
            else:
                fp+=1
        
    return np.array([[tp, fp], [fn, tn]])


def compute_precision(y_true, y_pred):
    """
    Function: compute_precision
    Invoke confusion_matrix() to obtain the counts
    """
    matrix=confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = (0,0), (0,1), (1,0), (1,1)
    
    return matrix[tp]/(matrix[tp]+matrix[fp])


def compute_recall(y_true, y_pred):
    """
    Function: compute_recall
    Invoke confusion_matrix() to obtain the counts
    """
    matrix=confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = (0,0), (0,1), (1,0), (1,1)

    return matrix[tp]/(matrix[tp]+matrix[fn])


def compute_accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    Invoke the confusion_matrix() to obtain the counts
    """
    matrix=confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = (0,0), (0,1), (1,0), (1,1)

    return (matrix[tp]+matrix[tn])/(matrix.sum())

y_true = np.array([0,  1,  1,  1,  0,  0,  0,  0,  0,  0])
y_pred = np.array([0,  0,  1,  1,  1,  1,  0,  0,  0,  1])
print(compute_accuracy(y_true, y_pred))

a=[]
a.append(2)
a.append(3)
print(type(np.array(a)))
def _compute_distances( X, x):
        '''
        Private function to compute distances. 
        Compute the distance between x and all points in X
    
        Parameters
        ----------
        x : a vector (data point)
        '''
        distances=[]

        for i in X:
            sum=0

            for j, k  in zip(x,i):
                diff_squared=(j-k)**2 
                sum=sum+diff_squared
            distance=np.sqrt(sum)
            distances.append(distance)


        return distances


mat = np.array([[1,21,3],[5,4,2],[56,12,4]])
print(mat[mat[:,1].argsort()])
#print(mat_sort)

a=np.array([[2],[3]])
b=np.array([[4],[5]])
c=np.concatenate((a,b),axis=1)
#print(c[:2,0])

a=" banananana      \t      x "
print(a.rstrip().split("\t")[0])

file_name_data= "data/part1/train/matrix_mirna_input.txt"
file_name_labels="data/part1/train/phenotype.txt"
id_data=[]
X=[]
this=list_ = open(file_name_data).read().split("\t")
print(type(this))


with open(file_name_data, 'r') as f_in:
    # Create a dictionary of lists. Key to the dictionary is the group name
    dict_doc = {}
    for line in f_in:
        # Remove the trailing newline and separate the fields
        parts = line.rstrip().split("\t")

        # If the group does not exist in the dictionary, create it
        if not parts[0] in dict_doc and parts[0]!='patientId':
            # Use the group name as key. Initialize list
            dict_doc[parts[0]] = []
            dict_doc[parts[0]].append(parts[1:])

with open(file_name_labels, 'r') as f_in:
    # Create a dictionary of lists. Key to the dictionary is the group name
    dict_doc_labels = {}
    for line in f_in:
        # Remove the trailing newline and separate the fields
        parts = line.rstrip().split("\t")

        # If the group does not exist in the dictionary, create it
        if not parts[0] in dict_doc_labels and parts[0]!='patientId':
            # Use the group name as key. Initialize list
            dict_doc_labels[parts[0]] = []
            dict_doc_labels[parts[0]].append(parts[1])

#extract values for X
X= np.array([val for val in dict_doc.values()],dtype=float)
X=np.reshape(X, (X.shape[0], X.shape[2]))


#initialise vector for the labels 
y=[]
for key, val in dict_doc.items():
    
    if dict_doc_labels[key]==['+']:
        y.append(1)
    else:
        y.append(0)
y=np.array(y)
PERF_METRICS = ["accuracy", "precision", "recall"]
print(PERF_METRICS[0],PERF_METRICS[1],\
    PERF_METRICS[2])

#this = np.loadtxt(file_name_data, skiprows=1, delimiter="\t" ,usecols=range(start=0)
#print(this)

X=np.genfromtxt("data/part2/train/tumor_info.txt", delimiter='\t', dtype=float)
print(np.count_nonzero(X[X[:,4]==2][:,0]==1))
print(len(X[X[:,4]==2][:,0]))
print(np.unique(X[:,4]))
a=['2', '4']
print(a[0])
print("no of 1 in class 2")
print(np.count_nonzero(X[X[:,4]==2][:,0]==1))
print(X[X[:,4]==2][:,0])
print(len(X[X[:,4]==2][:,0]))
print("no of samples in class 2")
class_2=len(X[X[:,4]==2])
print(class_2)
print("no of 1 in total")
print(np.count_nonzero(X[:,0]==1))
print("clump is ")
clump=np.count_nonzero(X[X[:,4]==2][:,0]==1)/(class_2-np.count_nonzero(np.isnan(X[X[:,4]==2][:,0])))
print(clump)
print(X)
print(X[:2,:])
print(X[2-1,:])

#%%
