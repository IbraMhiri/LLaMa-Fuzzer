from os import listdir, path
import pandas as pd

XMLS_SAMPLES_FOLDER = "$samples_folder"

dirs = listdir(XMLS_SAMPLES_FOLDER)

normal_xmls = []
malicious_xmls = []
for dir in dirs:
    dir_path = path.join(XMLS_SAMPLES_FOLDER, dir)
    print(f"Scanning directory {dir_path}")
    files=listdir(dir_path)
    #Process xml files
    for file in files:
        file_path=path.join(XMLS_SAMPLES_FOLDER, dir, file)
        with open(file_path, 'r') as fp:
            xml = fp.read()
            try:
                effective_xml = xml[xml.index("<"):]
                effective_xml = effective_xml[:effective_xml.rindex(">")+1]
            except Exception:
                print(f"error at {file}")
                exit()
            if dir == 'exploitDB':
                malicious_xmls.append(effective_xml)
            else:
                normal_xmls.append(effective_xml)

#Save xml text
df_normal = pd.DataFrame({'xml': normal_xmls})
df_malicous = pd.DataFrame({'xml': malicious_xmls})

df_normal = df_normal.set_index('xml')
df_normal.to_csv('./normal.csv', header=True)

df_malicous = df_malicous.set_index('xml')
df_malicous.to_csv('./malicious.csv', header=True)

# json_xmls = json.dumps(xmls)
# with open("train.json", "w") as outfile:
#     outfile.write(json_xmls)