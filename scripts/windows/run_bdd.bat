@echo off

setlocal

set script_dir=%~dp0
cd %script_dir%\..\..
set project_dir=%cd%

set images_dir=tests\bdd\images
set labels_dir=tests\bdd\labels
set output_dir=tests\bdd\tfrecord

set num_threads=%NUMBER_OF_PROCESSORS%

python -m cProfile %project_dir%\create_bdd_tf_record.py %images_dir% %labels_dir% %output_dir% %num_threads%

endlocal
