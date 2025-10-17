def liste(list):
    i = 0
    while i < len(list):
        before = list[i]
        after = list[i + 1]
        if before >  after and after < before :
            missing_number = before + 1
            break
        i += 1
    return missing_number 
lst= [1,3,4]
print(liste(lst))