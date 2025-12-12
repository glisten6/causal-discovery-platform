# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    pos = None
    flag = 0

    def reorderList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: None Do not return anything, modify head in-place instead.
        """

        if head is None or head.next is None:
            return head

        self.pos = head
        self.traceback(head)

    def traceback(self, root):

        if root is None:
            return
        self.traceback(root.next)

        if self.flag == 1:
            return

        if self.pos == root or self.pos.next == root:
            self.flag = 1
            print(111)
            return

        root.next = None
        t = self.pos.next
        # print(self.pos.val)
        self.pos.next = root
        root.next = t
        self.pos = t


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
# head.next.next.next.next = ListNode(4)
print(Solution().reorderList(head))


while head.next is not None:
    print(head.val)
    head = head.next