#!/bin/bash

# set dirs for data and caffe scripts
root_dir=$HOME/data/robot_dataset_extended2
train_dir=nhg
val_dir=weinholdbau
img_dir=Images
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
  if [ $dataset == "train" ]
  then
    dataset_file=$root_dir/$train_dir/$img_dir/$dataset.txt
  else
    dataset_file=$root_dir/$val_dir/$img_dir/$dataset.txt
  fi

  # tmp file for image paths
  img_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_file
  # add path
  if [ $dataset == "train" ]
  then
    sed -i "s/^/$train_dir\/$img_dir\//g" $img_file
  else
    sed -i "s/^/$val_dir\/$img_dir\//g" $img_file
  fi
  
  
  # tmp file for label/annotation paths
  label_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_file
  #sed -i -r 's/\s+\S+$//' $label_file
   if [ $dataset == "train" ]
  then
    sed -i "s/^/$train_dir\/$anno_dir\//g" $label_file
  else
    sed -i "s/^/$val_dir\/$anno_dir\//g" $label_file
  fi
  sed -i "s/.jpg/.xml/g" $label_file

  paste -d' ' $img_file $label_file >> $dst_file
  
  # Generate image name and size infomation.
  #if [ $dataset == "val" ]
  #then
  $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  #fi
  
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
