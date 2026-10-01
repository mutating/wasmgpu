#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t factorial(uint32_t n) {
    if (n < 2) return 1;
    return n * factorial(n - 1);
}

static double transform(double x) { return x * 1.25 + 0.5; }
static double (*volatile callback)(double) = transform;

__attribute__((export_name("process")))
double process(int n) {
    double *values = malloc((size_t)n * sizeof(double));
    if (values == NULL) return -1.0;
    double result = 0;
    for (int i = 0; i < n; ++i) {
        values[i] = callback((double)i);
        result += values[i];
    }
    free(values);
    return result + (double)factorial((uint32_t)n % 12);
}

__attribute__((export_name("file_process")))
int file_process(void) {
    FILE *input = fopen("numbers.txt", "r");
    if (input == NULL) return -1;
    char line[80];
    double sum = 0;
    while (fgets(line, sizeof(line), input)) sum += strtod(line, NULL);
    fclose(input);
    FILE *output = fopen("result.txt", "w");
    if (output == NULL) return -2;
    fprintf(output, "sum=%.3f\n", sum);
    fclose(output);
    printf("processed\n");
    return (int)(sum * 1000);
}
