#!/bin/bash

# Execute the find and copy command
find "CCE_NLI_real" -type f -not -path '*/\.*' -not -path '*/__*/*' -exec cp --parents {} /tutorial \;

echo "Files copied successfully from '$fldr' to /tutorial"
