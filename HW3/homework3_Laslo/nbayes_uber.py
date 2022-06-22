'''
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Calculates the psoterior probability P(N|D) for all valid values of N,
where N<=1000.

Author: Ana Daria Laslo

'''


N_max=1000
D=60 

#Calculating the posterior probability reequires the computing of three terms:
    #1: likelihood - depends on N
    #2: prior - does not ndepend on N
    #3: evidence - does not depend on N
#Note: probabilities for N<60 will not be considered as those are 0




################ Calculating the terms independent of N ########################

#calculate the prior probability, which is the same for every number
prior=1/N_max

#computing the evidence which does not change depending on N
#this is equal to the sum of the joint probabilities P(D,N), from D to N_max
#P(D,N)=P(D|N)*P(N)

#initialising this term to 0
evidence=0
#computing the evidence 
for N in range(D, N_max+1):
    term= (1/N_max)*(1/N)
    evidence=evidence+term




############## Computing the posterior probability, identifying the maximum #############

#initialising maximum probability and N for which it is attained to 0
max_prob=0
N_max_prob=0

for N in range(D, N_max+1):
    #calculate P(D|N) for every N
    likelihood= 1/N
    
    #the prior has been calculated before, this term does not change with N
    #the evidence has been calculated before, this term does not change with N

    #calculating the posterior probability 
    probability= likelihood*prior/evidence

    print("The prior probability for ", str(N), " is ", str(probability))

    if probability>max_prob:
        max_prob=probability
        N_max_prob=N

#printing the maximum probability and N for which it is attained 
print("The maximum probability ", str(max_prob), "is attained when N is ", str(N_max_prob))



    