Simple function to convert the results of a part in an Abaqus ODB file into a numpy array format that makes it easy to use for Deep Learning purposes. 
This must be run using the Abaqus CAE kernel. To also keep things simple we only use packages that are a part of native abaqus namely numpy and any
regular packaged python libraries.

Essentially results in ODB files are stored in a very OOP manner meaning you can't simply access results in an array like fashion.


Two formatting modes are available:
- Compact: All requested outputs are stored in a single large array of shape `[number Frames in step,number of nodes in Part,total number of output requests)]`
  This is memory efficient as we don't repeat nodal coordinates or defined parameters.  and suited for Deep Geometric Learning (e.g. Deep Graph networks)
- Tabular: Express the output as a csv like table as a 2D array of shape `[ (num Frames* number of nodes), (coords size + 1 + number of output requests)]`.
  This is less memory efficient as you are repeating parameters/co-ordinates of nodes. This format is suitable for paramaters based field networks
  (e.g. inputs to the network are (x,y,z,t) and any defined parameters p ) 
