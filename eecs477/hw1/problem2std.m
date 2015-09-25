# Problem 2, but in standard form
c = [-3, 5, 0, 0]';
A = [4, 5, -1, 0;
     6, -6, 0, 0;
     1, 8, 0, 1];
b = [3, 7, 20]';
lb = [0, 0, 0, 0];
ub = [];
ctype = "SSS";
vartype = "CCCC";
sense = 1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp(xmax);
