#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  int *p2 = (int*)aligned_alloc(1024, 1024*sizeof *p2);
  printf("1024-byte aligned addr: %p\n", (void*)p2);
  free(p2);
}
