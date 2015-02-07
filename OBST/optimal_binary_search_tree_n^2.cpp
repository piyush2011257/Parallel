// Dynamic Programming code for Optimal Binary Search Tree Problem
/* http://www.geeksforgeeks.org/dynamic-programming-set-24-optimal-binary-search-tree/

Optimal Substructure:
The optimal cost for freq[i..j] can be recursively calculated using following formula.
optCost(i, j) = \sum\limits_{k=i}^j freq[k] + \min\limits_{r=i}^k [ optCost(i, r-1) + optCost(r+1, j) ]

We need to calculate optCost(0, n-1) to find the result.

The idea of above formula is simple, we one by one try all nodes as root (r varies from i to j in second term). When we make rth node as root, we recursively calculate optimal cost from i to r-1 and r+1 to j.
We add sum of frequencies from i to j (see first term in the above formula), this is added because every search will go through root and one comparison will be done for every search.

Complexity is O(n^3). Knuth had observed that root[i, j - 1] <= root[i, j] <= root[i + 1, j], so by doing some modification to the original algorithm we get O(n^2)

for k = i to j
with
for k = root[i, j - 1] to root[i + 1, j]
*/

# include<ctime>
#include <cstdio>
#include <cstring>
#include<algorithm>
#include<limits.h>
using namespace std;
int n;

/* A Dynamic Programming based function that calculates minimum cost of a Binary Search Tree- O(n^3) */
int optimalSearchTree(int *freq, int *sum_frwd)
{	int cost[n+3][n+3], root[n+3][n+3];
	memset(cost, 0, sizeof(cost[0][0])*(n+3)*(n+3));
	memset(root, 0, sizeof(root[0][0])*(n+3)*(n+3));
	for ( int i=1; i<=n ; i++ )					// can be merged with the i/p
	{	cost[i][i]=freq[i];
		root[i][i]=i;
	}
	
	for ( int i=n; i>0; i-- )
	{	for ( int j=i+1; j<=n; j++ )
		{	int tmp=INT_MAX, tcost;				// limits.h
			for ( int k=root[i][j-1]; k<=root[i+1][j]; k++)	// observation by knuth
			{	tcost = cost[i][k-1] + cost[k+1][j];
				if ( tcost <= tmp )
				{	tmp= tcost;
					root[i][j]=k;
				}
			}
			cost[i][j]= sum_frwd[j] - sum_frwd[i-1] + tmp;
		}
	}
	
	/*
	for ( int i=1; i<=n+1; i++)
	{	for ( int j=1; j<=n+1;j++ )
			printf("%d %d\t", cost[i][j], root[i][j]);
		printf("\n");
	}*/
	return cost[1][n];
}

int main()
{	scanf("%d",&n);
	int freq[n+3], keys[n+3],sum_frwd[n+3];
	memset(sum_frwd, 0, sizeof(sum_frwd[0])*n+3);
	for ( int i=1; i<=n; i++ )
	{	//scanf("%d %d",keys+i, freq+i);
		freq[i]= rand()%100;//scanf("%d",freq+i);
		sum_frwd[i] = sum_frwd[i-1]+freq[i];
	}
	/*
	for ( int i=1; i<=n; i++ )
		printf("%d %d\n",freq[i], sum_frwd[i]);
	*/
	printf("time=%.3lf sec.\n",(double) (clock())/CLOCKS_PER_SEC);		// total time till this place
	printf("Cost of Optimal BST is %d ", optimalSearchTree(freq, sum_frwd));
	printf("time=%.3lf sec.\n",(double) (clock())/CLOCKS_PER_SEC);		// total time till this place
	return 0;
}


/*
Refer Karumanchi

f(i,j)= sum(freq[i->j]) + min ( limit k=i -> j   f(i,k-1) + f(k+1,j ) )	i<=j k+1<=j i<=k-1
this sum(freq[i->j]) due to addition of root and one more increase in depth.
filling of elements order
	1 2 3 4
       1-
       2  -
       3    -
       4      -  
order
34
23, 24
12, 13, 14

*/
