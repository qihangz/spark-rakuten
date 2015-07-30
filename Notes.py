Use Spark MLlib through PySpark on YARN
Note: the follwing steps are verified in Spark 1.2 and Spark 1.3
Steps:
0) Package the numpy into egg otherwise there will be Numpy Import Error.
0.1) Download the numpy (version>=1.4) and extract it to somewhere (e.g., $HOME/tools/numpy)
0.2) Go to the $HOME/tools/numpy and run  "python setupegg.py bdist_egg". We can find the egg file in the folder "$HOME/tools/numpy/dist"
1) sc.stop() --> stop the default SparkContext
2) sc = SparkContext(environment={'PYTHON_EGG_CACHE':'/tmp'})  --> specify the cache folder to a place where we have write privilege. (The default one could be "/home/.****")
3)  sc.addPyFile("/home/tools/numpy/dist/numpy-1.9.2-py2.7-linux-x86_64.egg") --> attach the egg file, it will be distributed on YARN automatically
Finally, find some examples from MLlib page and try them.
More info on configuring spark on YARN can be found in the following link:
https://confluence.rakuten-it.com/confluence/display/HADOOP/Install+Spark+Client+In+C4000

===========================

Spark error: bigger than the maxResultSize
enlarge the maxResultSize!

===========================

To display hardware information
lspci -vnn

===========================

pylearn2
pylearn2.utils.exc.EnvironmentVariableError: PYLEARN2_VIEWER_COMMAND not defined
=> output the image instead
show_weights.py --out=weights.png cifar_grbm_smd.pkl

cause: if using 
python ~/Documents/pylearn2/pylearn2/scripts/show_weights.py cifar_grbm_smd.pkl
cannot load environment variable

===========================


