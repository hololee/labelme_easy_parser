<h2> How to use </h2>

<h4> This parser script make easy to parse 'labelme' json data.</h4>  
json file contains original image data, label, and polygon information.  
  
So this scripts help to make label from json_file.  
  
  
There are three options.  

* json_path : `labelme` json files path.
* origin_path : original images saved path.
* target_semantic_path : semantic segmentation label images saved path.
* target_instance_path : instance segmentation label images saved path.

**Put `json_path` is necessary.**

```
python lmp_execute.py --json_path="{json_file_path}" 
                      --origin_path="{path want to put original image}" 
                      --target_semantic_path="{path want to put semantic label image}" 
                      --target_instance_path="{path want to put instance label image}"
```

* I'll update this for object detection dataset.