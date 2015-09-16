c = [30, 20, 100, 90, 160]';
A = [5, 10, 20, 30, 40];
b = [60]';
lb = [0, 0, 0, 0, 0];
ub = [1, 1, 1, 1, 1];
ctype = "U";
vartype = "CCCCC";
sense = -1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp(xmax);
