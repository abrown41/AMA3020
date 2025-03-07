# AMA3020 utilities

There are two utilities here

## assign.py

This tool reads a csv file containing the students ranked preferences for the projects, and then assigns the projects based on the Hungarian algorithm.

you can see the options by running 

```bash
python assign.py --help
```

The format of the csv file containing the student preferences is currently defined based on the tool at simpleassign.net, but we should move away from that and use either a canvas quiz or an MS form.

Running the assign tool creates both a csv file and an html file containing the allocation. The html can be copy-pasted into a canvas page and it will display a table with the allocations. 

## generate_solo.py

The way this module has been run has been to provide a list of projects to students which can be taken either for the pairs or solo project assessment. Once the pairs projects are assigned, we have been removing those projects from the pool of possible projects for the solo. To produce a PDF showing just the available projects, you can use the generate_solo.py utility. It uses the csv output of `assign.py` to determine which projects have already been used. 

```bash 
python generate_solo.py -f <path to csv file> 
```

This will produce a pdf 'solo_projects.pdf' which can be uploaded to canvas.