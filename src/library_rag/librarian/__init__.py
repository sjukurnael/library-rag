"""
The librarian: you describe a topic, it recommends books for a classroom.

Replaced the old browsing agent, which could only see filenames and folder
paths. That agent's own system prompt had to apologise for it -- "You can only
see filenames and sizes. You have NOT read these books." For an indexed book
that is no longer true, and the difference is the whole point: this one opens a
book before recommending it and quotes the passage that justifies the pick.
"""
