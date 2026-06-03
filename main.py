nums=[[2],[3,4],[6,5,7],[4,1,8,3]]

for num in nums:
    num.sort()

print(nums)
# then it is sorted so we need to add the first element of each row
res=0
for i in range(len(nums)):
    res+=nums[i][0]
print(res)
# Wrong Answer
# 14 / 46 testcases passed


# Input
# triangle =
# [[-1],[2,3],[1,-1,-3]]

# Use Testcase
# Output
# -2
# Expected
# -1

