import sys
mess=[]
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        mess.append(values)

for i in range(n):
	mess[i][-1]=i+1

isinline=False
List=[]
List.append(mess[0])
for i in range(1,n):
	for j in range(len(mess[i])):
		for k in range(len(List)):
			if mess[i][j]-1 in List[k]:
				isinline=True
				List[k]=List[k]+mess[i]
	if isinline==False:
		List.append([])

print(len(List))



