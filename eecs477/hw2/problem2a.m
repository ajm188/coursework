c = [1, 2]';
A = [1, 1;
     6, -3;
     5, 0;
     0, 6];
b = [5, 3, 24, 9]';
lb = [0, 0];
ub = [];
ctype = "UUUU";
vartype = "CC";
sense = -1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp(xmax);
disp(fmax);
disp(status);
