
Logistic Regression
====================

ロジスティック回帰における予測値は以下の式で表される。

..math::

  y\hat = \sigma(xw + b)

ここで :math:`\sigma` はLogistic sigmoid functionで以下で表される。

..math::

  \sigma(x) = \frac{1}{1 + e^{-x}}

損失関数は予測値 :math:`y\hat` と実際の値 :math:`y` との差で表される。この損失関数を最小化する :math:`w, b` を求める。それを求めるためには勾配法を用いる。勾配法は例えば以下の3つなどがある。

* Gradient Descent (勾配降下法)
* Stochastic Gradient Descent (確率的勾配降下法)
* Minibatch SGD / MSGD (ミニバッチ確率的勾配降下法)


	
	
