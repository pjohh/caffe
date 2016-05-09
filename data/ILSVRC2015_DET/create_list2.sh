#!/bin/bash

# set dirs for data and caffe scripts
root_dir=$HOME/data/ILSVRC2015/
sub_dir=ImageSets/DET
data_dir=Data
anno_dir=Annotations
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# for train and val do
for dataset in train val
do
  # destination file is path/trainval.txt or test.txt
  dst_file=$bash_dir/$dataset.txt
  # remove if it exists
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  
  # load trainval.txt / test.txt
  echo "Create list for $dataset..."
  dataset_file=$root_dir/$sub_dir/$dataset.txt

  # tmp file for image paths
  img_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_file
  # remove last collumn
  sed -i -r 's/\s+\S+$//' $img_file
  # add path
  sed -i "s/^/$data_dir\/DET\/$dataset\//g" $img_file
  # add .jpg
  sed -i "s/$/.JPEG/g" $img_file
  
  # tmp file for label/annotation paths
  label_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_file
  sed -i -r 's/\s+\S+$//' $label_file
  sed -i "s/^/$anno_dir\/DET\/$dataset\//g" $label_file
  sed -i "s/$/.xml/g" $label_file
  # no annotations for ilsvrc2013_train_extra0 --> use dummy annotation
  #sed -i "s/extra0.*/negative\/dummy_annotation.xml/" $label_file
  
  paste -d' ' $img_file $label_file >> $dst_file
  
  # remove all negative examples
  sed -i "/extra.*/d" $dst_file

  # Generate image name and size infomation.
  if [ $dataset == "val" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi
  
  # Shuffle train file.
  if [ $dataset == "train" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi

  rm -f $label_file
  rm -f $img_file
done
