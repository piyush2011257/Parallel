#include<stdio.h>
#include<stdlib.h>

void test1( int *a )
{	a= (int*)malloc(sizeof(int));			// declared here!! so local scope
	printf("%p\n",a);
}

void test2 ( int **a)
{	a= (int**)malloc(sizeof(int*));
	printf("%p\n",a);
}

int main()
{	int **a;
	test2(a);					// dangling pointer!
	printf("%p\n\n",a);
	int *b;
	test1(b);
	printf("%p\n\n", b);
	void *a1;
	*a1= 2;
	return 0;
}
