"""
1. Question
https://rosalind.info/problems/iprb/

in a given population, where k,m,n represent the total population k +m +n ogarnims /
- k individuals are homozygous dominent for the brown eye color alle
- m are Heterozygous for the brown eye color allet 
- n are homozygous recessive(show blue eye color)

What is The probability that two randomly selected mating organisms will produce an individual possessing a dominant allele(*) 
(and thus displaying the dominant phenotype). Assume that any two organisms can mate.

Note :
- Homozygous dominent mean they have two dominant allele (AA) 
- Heterozygous mean they have one dominant allele (A) and one recessive allele (a) 
- homozygous recessive mean they have two recessive allele (aa) 
The individual possessing a dominant allele(H) is either homozygous dominent or Heterozygous(*) 

The probability individual possessing a dominant allele(A) is either homozygous dominent or Heterozygous = 1 - The probability individual possessing only homozygous recessive 
Take a look at table (**) for more details

2. Solution
let's suppose:
- Homozygous dominent is 'd' : which contain two dominant allele (AA), there is overall k ogarnims
- Heterozygous is 'h' : which have one dominant allele (A) and one recessive allele (a), there is overall m ogarnims
- homozygous recessive is 'r' : which have two recessive allele (aa), there is overall n ogarnims

About the offspring is completely homozygous recessive when(Take a look at table (**) for more details): 

Case 1. if Heterozygous mate Heterozygous(h vs h)
	 A	a
A	AA	Aa
a	Aa	aa
=> The probability of offspring being homozygous recessive (aa) is 0.25
Case 2. if Heterozygous mate homozygous recessive (h vs r)
	A	a
a	Aa	aa
a	Aa	aa
=> The probability of offspring being homozygous recessive (aa) is 0.5
Case 3. if homozygous recessive meet Heterozygous (r vs h)
	 a  a	
A	Aa	Aa
a	aa	aa
=> The probability of offspring being homozygous recessive (aa) is 0.5

Case 4. if homozygous recessive meet homozygous recessive (r vs r)
     a  a 
a   aa  aa
a   aa  aa
=> The probability of offspring being homozygous recessive (aa) is 1

Sumary:
Let X = the r.v. associated with the first person randomly selected
Let Y = the r.v. associated with the second person randomly selected without replacement
Then:
k = f_d => p(X=d) = k/a => p(Y=d| X=d) = (k-1)/(a-1) ,
                           p(Y=h| X=d) = (m)/(a-1) ,
                           p(Y=r| X=d) = (n)/(a-1)
                           
m = f_h => p(X=h) = m/a => p(Y=d| X=h) = (k)/(a-1) ,
                           p(Y=h| X=h) = (m-1)/(a-1)
                           p(Y=r| X=h) = (n)/(a-1)
                           
n = f_r => p(X=r) = n/a => p(Y=d| X=r) = (k)/(a-1) ,
                           p(Y=h| X=r) = (m)/(a-1) ,
                           p(Y=r| X=r) = (n-1)/(a-1)
Now the joint would be:(**)
                            |    offspring possibilites given X and Y choice
-------------------------------------------------------------------------
X Y |  P(X,Y)               |   d(dominant)     h(hetero)   r(recessive)
-------------------------------------------------------------------------
d d     k/a*(k-1)/(a-1)     |    1               0           0
d h     k/a*(m)/(a-1)       |    1/2            1/2          0
d r     k/a*(n)/(a-1)       |    0               1           0
                            |
h d     m/a*(k)/(a-1)       |    1/2            1/2          0
h h     m/a*(m-1)/(a-1)     |    1/4            1/2         1/4 (case 1)
h r     m/a*(n)/(a-1)       |    0              1/2         1/2 (case 2)
                            |
r d     n/a*(k)/(a-1)       |    0               0           0
r h     n/a*(m)/(a-1)       |    0               1/2        1/2 (case 3)
r r     n/a*(n-1)/(a-1)     |    0               0           1  (case 4)

Here what we don't want is the element in the very last column where the offspring is completely recessive.
so P = 1 - P(offspring=recessive)


- prob_kk(two type d mate) : (k / total_population) * ((k - 1) / (total_population - 1))
- prob_km(type d mate type h) : (k / total_population) * (m / (total_population - 1)) + (m / total_population) * (k / (total_population - 1))
- prob_kn(type d mate type r) : (k / total_population) * (n / (total_population - 1)) + (n / total_population) * (k / (total_population - 1))
(1) prob_mm(two type h mate) : (m / total_population) * ((m - 1) / (total_population - 1))    
(2) prob_mn(type h mate type r) : (m / total_population) * (n / (total_population - 1)) + (n / total_population) * (m / (total_population - 1))
(3) prob_nm(type r mate type h) : (n / total_population) * (m / (total_population - 1)) + (m / total_population) * (n / (total_population - 1))  
(4) prob_nn(two type r mate) : (n / total_population) * ((n - 1) / (total_population - 1))
"""

def solution(k, m, n):
    total_population = k + m + n

    # Calculate the probabilities for each scenario
    # prob_kk = (k / total_population) * ((k - 1) / (total_population - 1))
    # prob_km = (k / total_population) * (m / (total_population - 1)) + (m / total_population) * (k / (total_population - 1)) 
    # prob_kn = (k / total_population) * (n / (total_population - 1)) + (n / total_population) * (k / (total_population - 1))
    prob_mm = (m / total_population) * ((m - 1) / (total_population - 1))   #(1)
    prob_mn = (m / total_population) * (n / (total_population - 1)) #(2)
    prob_nm = (n / total_population) * (m / (total_population - 1)) #(3)
    prob_nn = (n / total_population) * ((n - 1) / (total_population - 1)) #(4)



    # Calculate the overall probability of producing an individual completely recessive(take a look at the table **)
    prob_offspring_recessive = prob_mm * 0.25 + prob_mn * 0.5 + prob_nm * 0.5  +  prob_nn * 1

    prob_homozygous_dominent_or_heterozygous = 1 - prob_offspring_recessive
    return prob_homozygous_dominent_or_heterozygous




# Test the function with sample inputs
k = 29
m = 48
n = 68
# the expected result is 0.598467
print(solution(k, m, n))  # Output the overall probability
