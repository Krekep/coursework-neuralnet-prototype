import project.console
from project import console

if __name__ == "__main__":
    print("exec sources directory")
    console.run_console()

"""
table ../train_data.csv
build-plot table0 --interval 0 4 --step 0.001
build-plot table0 --interval 0 3 --step 0.001
build-plot table0 --interval 0 2 --step 0.2
build-plot table0 --interval -6.3 6.3 --step 0.009
build-plot table0 --interval -12 12 --step 0.001

table ../train_data2.csv
build-plot table0 --interval 0 4 --step 0.001

save-to-file table0 C:/Users/Pavel/PycharmProjects/coursework/networks
load-from-file table0 C:/Users/Pavel/PycharmProjects/coursework/networks/table0.txt

table ../train_data_3d.csv
"""