#!/bin/bash
activate () {
    . venv/bin/activate
}
activate

#pdf_path=/home/timur/PycharmWork/classification_text/all_files
#pdf_done_path=/home/timur/PycharmWork/classification_text/done
pdf_path=/home/ruscon/IDP/classification/all_files
pdf_done_path=/home/ruscon/IDP/classification/done
mkdir "${pdf_done_path}"
echo "$pdf_path"

cache="${pdf_path}"/cache
if [ ! -d "$cache" ]; then
  mkdir "${cache}"
fi

csv_path="${pdf_path}"/csv
if [ ! -d "$csv_path" ]; then
  mkdir "${csv_path}"
fi

json_path="${pdf_path}"/json
if [ ! -d "$json_path" ]; then
  mkdir "${json_path}"
fi

done_path="${pdf_path}"/done
if [ ! -d "$done_path" ]; then
  mkdir "${done_path}"
fi


find "${pdf_path}" -maxdepth 1 -type f \( -name "*.pdf*" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file
do
  basename_file=$(basename "$file")
  echo "$basename_file"
  if [[ "${file}" == *"error_"* ]];
  then
    echo "Contains an error in ${file}"
    continue
  fi

	mime_type=$(file -b --mime-type "$file")
  echo "'${file} - ${mime_type}'"

  if [[ ${mime_type} = "application/pdf" ]]
  then
    python3 -c "import work_with_multi_pages; work_multi_pages = work_with_multi_pages.WorkMultiPages('${file}', '${pdf_path}/cache/${basename_file}', '${pdf_path}/dir_classific'); work_multi_pages.len_files_in_file()"
    pdfimages -all "$file" "$pdf_path/cache/$basename_file"
    python3 -c "import work_with_multi_pages; work_multi_pages = work_with_multi_pages.WorkMultiPages('${file}', '${pdf_path}/cache/${basename_file}', '${pdf_path}/dir_classific'); work_multi_pages.len_files_in_cache()"
  else
    echo "ERROR: unsupported format ${mime_type}"
    mv "${file}" "${pdf_path}/error_$(basename "${file}")"
    continue
  fi

  for file_jpg in $pdf_path/cache/*
  do
    classification_name=$(python3 work_with_multi_pages.py "$file_jpg" "$pdf_path/cache/$basename_file" "$pdf_path/dir_classific")
    jpg_name="${pdf_path}/dir_classific/$classification_name/$(basename "${file_jpg}")"
    base_path_yaml="config_yaml/$classification_name"
    config_yaml_file="${base_path_yaml}/$classification_name.yml"
    scripts_for_validations_and_postprocessing="${base_path_yaml}/$classification_name.py"
    python3 ocr_table.py "$jpg_name" "$pdf_path/json" "$pdf_path/csv" "${config_yaml_file}" "${scripts_for_validations_and_postprocessing}"
    if [ $? -eq 0 ]
    then
      mkdir "${done_path}/$classification_name"
      mv "${jpg_name}" "${done_path}/$classification_name/$(basename "${jpg_name}")"
    else
      mv "${jpg_name}" "${pdf_path}/dir_classific/$classification_name/error_$(basename "${jpg_name}")"
    fi
  done

#  find "${pdf_path}/cache" -maxdepth 1 -type f \( -name "*.jpg" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file_jpg
#  do
#    echo "fdhb"
#    python3 work_with_multi_pages.py "$file_jpg" "$pdf_path/cache/$basename_file" "$pdf_path/dir_classific"
#
#    jpg_name="${pdf_path}/dir_classific/port/$(basename "${file_jpg}")"
#    base_path_yaml="config_yaml/port"
#    config_yaml_file="${base_path_yaml}/port.yml"
#    scripts_for_validations_and_postprocessing="${base_path_yaml}/port.py"
#    echo "${scripts_for_validations_and_postprocessing}"
#    python3 ocr_table.py "$file_jpg" "$pdf_path/json" "$pdf_path/csv" "${config_yaml_file}" "${scripts_for_validations_and_postprocessing}"
#    if [ $? -eq 0 ]
#    then
#      mkdir "${done_path}/port"
#      mv "${jpg_name}" "${done_path}/port/$(basename "${jpg_name}")"
#    else
#      mv "${jpg_name}" "${pdf_path}/dir_classific/port/error_$(basename "${jpg_name}")"
#    fi
#  done

#	find "${pdf_path}/dir_classific/port" -maxdepth 1 -type f \( -name "*.jpg" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file_jpg
#	do
#	  jpg_name="${pdf_path}/dir_classific/port/$(basename "${file_jpg}")"
#	  base_path_yaml="config_yaml/port"
#	  config_yaml_file="${base_path_yaml}/port.yml"
#	  scripts_for_validations_and_postprocessing="${base_path_yaml}/port.py"
#	  echo "${scripts_for_validations_and_postprocessing}"
#	  python3 ocr_table.py "$file_jpg" "$pdf_path/json" "$pdf_path/csv" "${config_yaml_file}" "${scripts_for_validations_and_postprocessing}"
#    if [ $? -eq 0 ]
#    then
##      true
#      mkdir "${done_path}/port"
#      mv "${jpg_name}" "${done_path}/port/$(basename "${jpg_name}")"
#    else
#      mv "${jpg_name}" "${pdf_path}/dir_classific/port/error_$(basename "${jpg_name}")"
#    fi
#	done
#
#	find "${pdf_path}/dir_classific/contract" -maxdepth 1 -type f \( -name "*.jpg" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file_jpg
#	do
#	  jpg_name="${pdf_path}/dir_classific/contract/$(basename "${file_jpg}")"
#	  base_path_yaml="config_yaml/contract"
#	  config_yaml_file="${base_path_yaml}/contract.yml"
#	  scripts_for_validations_and_postprocessing="${base_path_yaml}/contract.py"
#	  python3 ocr_table.py "$file_jpg" "$pdf_path/json" "$pdf_path/csv" "${config_yaml_file}" "${scripts_for_validations_and_postprocessing}"
#    if [ $? -eq 0 ]
#    then
##      true
#      mkdir "${done_path}/contract"
#      mv "${jpg_name}" "${done_path}/contract/$(basename "${jpg_name}")"
#    else
#      mv "${jpg_name}" "${pdf_path}/dir_classific/contract/error_$(basename "${jpg_name}")"
#    fi
#	done
#
#	find "${pdf_path}/dir_classific/line" -maxdepth 1 -type f \( -name "*.jpg" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file_jpg
#	do
#	  jpg_name="${pdf_path}/dir_classific/line/$(basename "${file_jpg}")"
#	  base_path_yaml="config_yaml/line"
#	  config_yaml_file="${base_path_yaml}/line.yml"
#	  scripts_for_validations_and_postprocessing="${base_path_yaml}/line.py"
#	  python3 ocr_table.py "$file_jpg" "$pdf_path/json" "$pdf_path/csv" "${config_yaml_file}" "${scripts_for_validations_and_postprocessing}"
#    if [ $? -eq 0 ]
#    then
##      true
#      mkdir "${done_path}/line"
#      mv "${jpg_name}" "${done_path}/line/$(basename "${jpg_name}")"
#    else
#      mv "${jpg_name}" "${pdf_path}/dir_classific/line/error_$(basename "${jpg_name}")"
#    fi
#	done
#
#	find "${pdf_path}/dir_classific/unknown" -maxdepth 1 -type f \( -name "*.jpg" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file_jpg
#	do
#	  jpg_name="${pdf_path}/dir_classific/unknown/$(basename "${file_jpg}")"
#	  config_yaml_file="config_yaml/unknown.yml"
#	  python3 ocr_table.py "$file_jpg" "$pdf_path/json" "$pdf_path/csv" "${config_yaml_file}"
#    if [ $? -eq 0 ]
#    then
#      mkdir "${done_path}/unknown"
#      mv "${jpg_name}" "${done_path}/unknown/$(basename "${jpg_name}")"
#    else
#      mv "${jpg_name}" "${pdf_path}/dir_classific/unknown/error_$(basename "${jpg_name}")"
#    fi
#	done

#mv "${file}" "${pdf_done_path}"
done

#rm -r "${pdf_path}/cache"











#pdf_path=/home/timur/PycharmWork/classification_text/all_files
#path_project=/home/timur/PycharmWork/classification_text
#mkdir "$path_project/cache"  # ?????????????? ?????????? ?? ??????????
#pdf_list="${pdf_path}/*"
#for pdf in $pdf_list
#
#do
#
#    b=$(basename "$pdf")
#    echo $b
#
#    python3 len_files_in_file.py "$pdf_path/$b" # ?????????????? ?? files_in_file.json ???????????????????? ?????????????? ?? # ?????????????????????????????? ??????????
#    pdfimages -all "$pdf" "$path_project/cache/$b" # ?????????????????? ???? ?????????????????????????? ?????????? ???? ???????????????????????????????? ??????????
#    python3 len_files_in_cache.py "$path_project/cache" # ???????????????????????? ?????????????????????? ?????????????????????? ????????????
#
#
#    python3 turn_img.py "$path_project/cache" "$path_project/turned_img"
##    python3 turn_small_degree.py "$path_project/turned_img" "$path_project/turned_img"
#
#
##    python3 load_model_for_classific.py "$path_project/turned_img" "$path_project/dir_classific"
##
##    python3 ocr_table.py "$path_project/dir_classific/port" "$path_project/json" "$path_project/csv"
#
#    rm -r $path_project/cache/*.jpg  # ?????????????? ??????
##    rm -r $path_project/dir_classific/port/*.jpg
#
#done
#
##rm -r "${path_project}/cache"
##rm -r "${path_project}/turned_img"

