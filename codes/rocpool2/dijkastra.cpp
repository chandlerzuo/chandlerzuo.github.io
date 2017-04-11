#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::IntegerMatrix dijkastra(NumericMatrix fpr, NumericMatrix tpr) {
  const int m = fpr.nrow(), n = fpr.ncol();
  NumericMatrix cost(m, n);
  bool conn_h[m - 1][n], conn_v[m][n - 1];
  //back track to find the path
  IntegerMatrix path(m + n - 1, 2);
  
  for(int i = 0; i < m; i ++) {
    for(int j = 0; j < n; j ++) {
      if(i < m - 1) {
        conn_h[i][j] = false;
      }
      if(j < n - 1) {
        conn_v[i][j] = false;
      }
    }
  }

  cost(0, 0) = fpr(0, 0) * tpr(0, 0) / 2;
  for(int i = 1; i < m; i ++) {
    cost(i, 0) = cost(i - 1, 0) + (fpr(i, 0) - fpr(i - 1, 0)) * (tpr(i, 0) + tpr(i - 1, 0)) / 2;
    conn_h[i - 1][0] = true;
  }
  for(int i = 1; i < n; i ++) {
    cost(0, i) = cost(0, i - 1) + (fpr(0, i) - fpr(0, i - 1)) * (tpr(0, i) + tpr(0, i - 1)) / 2;
    conn_v[0][i - 1] = true;
  }
  
  for(int i = 1; i < m; i ++) {
    for(int j = 1; j < n; j ++) {
      double val1 = cost(i - 1, j) + (fpr(i, j) - 
                         fpr(i - 1, j)) * (tpr(i, j) + tpr(i - 1, j)) / 2;
      double val2 = cost(i, j - 1) + (fpr(i, j) - 
                         fpr(i, j - 1)) * (tpr(i, j) + tpr(i, j - 1)) / 2;
      if(val1 < val2) {
        cost(i, j) = val2;
        conn_v[i][j - 1] = true;
      } else {
        cost(i, j) = val1;
        conn_h[i - 1][j] = true;
      }
    }
  }

  int i = m - 1, j = n - 1;
  while(i > 0 || j > 0) {
    path(i + j, 0) = i + 1;
    path(i + j, 1) = j + 1;
    if(i >= 1 && conn_h[i - 1][j]) {
      i --;
    } else {
      j --;
    }
  }
  path(0, 0) = 1;
  path(0, 1) = 1;

  return path;
}