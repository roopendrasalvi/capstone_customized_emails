def route(super_category, sub_category):
 
    if super_category == "category0":
        if sub_category == "birthday":
            return "Birthday Email Workflow"
        if sub_category == "work_anniversary":
            return "Work Anniversary Workflow"
 
    if super_category == "category1":
        if sub_category == "leave":
            return "Leave Approval Workflow"
        if sub_category == "meeting":
            return "Meeting Scheduling Workflow"
        if sub_category == "asset":
            return "Asset/IT Support Workflow"
        if sub_category == "approval":
            return "Approval Workflow"
 
    if super_category == "promotional":
        return "Promotional Email Workflow"
 
    return "General Information Workflow"
 