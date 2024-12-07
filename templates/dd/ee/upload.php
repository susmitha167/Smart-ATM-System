<?php

extract($_REQUEST);
$fn=$_FILES['file']['name'];
move_uploaded_file($_FILES['file']['tmp_name'],"upload/".$fn);
?>