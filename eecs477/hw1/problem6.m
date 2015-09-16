c = [1, 100]'; # varying the second value in this vector will vary the F/I ratio
A = [5, 10];
b = [9]';
lb = [0, 0];
ub = [1, 1];
ctype = "U";
sense = -1;

[x_c_max, f_c_max, status, extra] = glpk(c, A, b, lb, ub, ctype, "CC", sense);
[x_i_max, f_i_max, status, extra] = glpk(c, A, b, lb, ub, ctype, "II", sense);
disp(f_c_max);
disp(f_i_max);
