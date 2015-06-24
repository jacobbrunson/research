#Research

I am working full-time at the Human Intelligence and Language Technologies (HiLT) lab at the University of North Texas for the Summer of 2015.

We are creating an "I Spy" game with [NAO Robots](https://www.aldebaran.com/en/humanoid-robot/nao-robot) in order to teach them about objects around them. You can read more about the project [here.](http://hilt.cse.unt.edu/ispy.html)

You can find my contributions to the main codebase [here.](https://github.com/iamadamhair/ispy_python)


###segment.py
Segment.py is an initial attempt to take a video stream of the object field, and gather images of individual objects. This assumes objects of reasonable size on a plain white background.

###track.py
Track.py builds upon segment.py. It periodically gathers images of objects in the scene, and groups the images based on the object in the image. The result is many collections of images that can be used as training data for the system. Ideally this will be used as a way to gather more data in real time during games. Collections of images that are gathered can be studied to determine if there are new (and therefore unknown) objects on the field.