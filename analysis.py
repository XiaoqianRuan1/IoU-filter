from collections import defaultdict
import matplotlib.pyplot as plt

def read_text(txt_path):
    with open(txt_path,"r") as f:
        results = f.readlines()
    results = results[0]
    return results

def generate_data(results):
    results = results.split("[")
    final_results = []
    for i,result in enumerate(results):
        final_result = []
        result = result.split(",")
        for index,r in enumerate(result):
            if index%4==0:
                r = r.split("(")
                if len(r)>1:
                    fin1=int(r[2])
            if index%4==2:
                r = r.split("(")
                fin2 = float(r[1])
            if index%4==3:
                final_result.append((fin1,fin2))
        final_results.append(final_result)
    return final_results        

def data_analysis(final_results):
    for results in final_results:
        for result in results:
            if float(result[1])>0.95:
                print(result)

def write_lines(results):
    results = results[0]
    results = results.split(",")
    final_results = []
    for i,result in enumerate(results):
        if i%4==0:
            result = result.split("(")
            fin_1 = result[2]
        if i%4==2:
            result = result.split("(")
            fin_2 = result[1]
        if i%4==3:
            final_results.append((fin_1,fin_2))
    return final_results
    
def analyse_result(results):
    final_results = defaultdict(list)
    for result in results:
        key = int(result[0])
        value = float(result[1])
        final_results[key].append(value)
    return final_results

def plot_image(results):
    for key in results.keys():
        print(key)
        result = results[key]
        plt.bar(range(len(result)), result)
        plt.show()                    
        
if __name__=="__main__":
    results = read_text("results_new_0.txt")
    final_results = generate_data(results)
    data_analysis(final_results)