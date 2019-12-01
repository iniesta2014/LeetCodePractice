package come.example.qiaowenhao.leetcodepractices;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

public class LeetCodeActivity extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leetcode);
        int[] nums = {2,3,1,2,4,3};
        Log.d("iniesta", "onCreate: " + minSubArrayLen(7,nums));
    }

    // 349. Intersection of Two Arrays
    // ****
    public int[] intersection(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0, j = 0;
        Set<Integer> set = new HashSet<Integer>();
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                i++;
            } else if (nums1[i] > nums2[j]) {
                j++;
            } else {
                set.add(nums1[i]);
                i++;
                j++;
            }

        }
        int[] res = new int[set.size()];
        int index = 0;
        Iterator iterator = set.iterator();
        while (iterator.hasNext()) {
            res[index++] = ((Integer) iterator.next());
        }
        return res;
    }

    // 75. Sort Colors
    // ****
    public void sortColors(int[] nums) {
        int start = 0, end = nums.length - 1;
        for (int i = 0; i <= end; i++) {
            while (nums[i] == 2 && i < end) {
                swap(nums, i, end--);
            }
            while (nums[i] == 0 && i > start) {
                swap(nums, i, start++);
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[j];
        nums[j] = nums[i];
        nums[i] = temp;
    }


    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    // 61. Rotate List
    //****
    public ListNode rotateRight(ListNode head, int k) {
        int length = getLength(head);
        //注意边界条件
        if (length == 0|| k == 0 || k % getLength(head) == 0) {
            return null;
        }
        k = k % getLength(head);
        ListNode newHead = new ListNode(-1);
        newHead.next = head;
        ListNode slow = newHead;
        ListNode fast = newHead;
        int n = k;
        while (n-- > 0) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        newHead.next = slow.next;
        slow.next = null;
        fast.next = head;
        return newHead.next;
    }

    private int getLength(ListNode node) {
        int length = 0;
        while (node != null) {
            node = node.next;
            length++;
        }
        return length;
    }

    // 3. Longest Substring Without Repeating Characters
    //****
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Queue<Character> queue = new LinkedList<>();
        char[] chars = s.toCharArray();
        int length = Integer.MIN_VALUE;
        for (int i = 0; i < chars.length; i++) {
            while (queue.contains(chars[i])) {
                queue.poll();
            }
            queue.add(chars[i]);
            length = Math.max(length, queue.size());
        }
        return length;
    }

    // 11. Container With Most Water
    // **
    public int maxArea(int[] height) {
        int res = 0;
        int start = 0, end = height.length - 1;
        while (start < end) {
            res = Math.max(res, Math.min(height[start], height[end]) * (end - start));
            if (height[start] > height[end]) {
                end--;
            } else {
                start++;
            }
        }
        return res;
    }

    // 3Sum
    // ****
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3) {
            return res;
        }

        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i == 0 || (i > 0 && nums[i-1] != nums[i])) {
                int start = i + 1, end = nums.length - 1;
                while (start < end) {
                    if (nums[start] + nums[end] == -nums[i]) {
                        res.add(Arrays.asList(nums[i], nums[start], nums[end]));
                        while (start < end && nums[start + 1] == nums[start]) start++;
                        while (start < end && nums[end -1] == nums[end]) end--;
                        start++;
                        end--;
                    } else if (nums[start] + nums[end] < -nums[i]) {
                        start++;
                    } else {
                        end--;
                    }
                }
            }
        }
        return res;
    }

    // 19. Remove Nth Node From End of List
    // ***
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int length = getSize(head);
        if (length == 0 || length < n) {
            return head;
        }
        ListNode helper = new ListNode(-1);
        helper.next = head;
        ListNode fast = helper;
        ListNode slow = helper;
        while (n-- > 0) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = fast;
        return helper.next;
    }

    private int getSize(ListNode listNode) {
        int length = 0;
        while (listNode != null) {
            listNode = listNode.next;
            length++;
        }
        return length;
    }

    // 16. 3Sum Closest
    //***
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length < 3) {
            return 0;
        }
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[nums.length - 1];
        for (int i = 0; i < nums.length - 2; i++) {
            int start = i + 1;
            int end = nums.length - 1;
            while (start < end) {
                int tmp = nums[i] + nums[start] + nums[end];
                if (tmp < target) {
                    start++;
                } else if (tmp > target) {
                    end--;
                } else {
                    return tmp;
                }
                res = Math.abs(tmp - target) < Math.abs(res - target) ? tmp : res;
            }
        }
        return res;
    }

    public boolean hasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }


    // 142. Linked List Cycle II
    // ***
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        boolean hasCirlce = false;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                hasCirlce = true;
                break;
            }
        }
        if (!hasCirlce) {
            return null;
        }
        while (slow != head) {
            slow = slow.next;
            head = head.next;
        }
        return head;
    }

    // 86. Partition List
    // ***　注意是值传递并不是引用传递
    public ListNode partition(ListNode head, int x) {
        ListNode lower = new ListNode(-1);
        ListNode lower1 = lower;
        ListNode higher = new ListNode(-1);
        ListNode higher1 = higher;
        while (head != null) {
            if (head.val < x) {
                lower.next = head;
                lower = lower.next;
            } else {
                higher.next = head;
                higher = higher.next;
            }
            head = head.next;
        }
        higher.next = null;
        lower.next = higher1.next;
        return lower1.next;
    }

   // 80. Remove Duplicates from Sorted Array II
    //****
    public int removeDuplicates(int[] nums) {
        int count = 1, i = 0, j= 0;
        while (j < nums.length) {
           if (j - 1 >= 0 && nums[j] == nums[j- 1]) {
               count++;
               if (count <= 2) {
                   nums[i++] = nums[j++];
               } else {
                   j++;
               }
           } else {
               count = 1;
               nums[i++] = nums[j++];
           }
        }
        return i;
    }


    // 209. Minimum Size Subarray Sum
    // **** 注意边界条件，双重循环
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null) {
            return 0;
        }
        int slow = 0, fast = 0;
        int sum = 0;
        int res = Integer.MAX_VALUE;
        while (fast < nums.length) {
            sum += nums[fast];
            fast++;
            while (sum >=s) {
                res = Math.min(res, fast - slow);
                sum -= nums[slow];
                slow++;
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    // 24. Swap Nodes in Pairs
    // **** 理解递归
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = head.next;
        head.next = swapPairs(head.next.next);
        newHead.next = head;
        return newHead;
    }

    // 迭代，辅助指针，先局部翻转，再移动指针
    public ListNode swapPairs1(ListNode head) {
        ListNode helper = new ListNode(-1);
        helper.next = head;
        ListNode newHead = helper;
        while (head != null && head.next != null) {
            ListNode second = head.next;
            head.next = second.next;
            second.next = head;
            helper.next = second;
            helper = head;
            head = head.next;
        }
        return newHead.next;
    }

    // 143. Reorder List
    // ***** 链表翻转、链表插入、值传递、 注意边界条件
    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return;
        }
        ListNode middle = getMiddle(head);
        ListNode first = head;
        ListNode second = middle.next;
        middle.next = null;
        mergeList(first, reverse(second));
    }
    // 考虑奇数和偶数
    public ListNode getMiddle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    //　迭代
    private ListNode reverse(ListNode head) {
        ListNode pre = head;
        ListNode cur = head.next;
        ListNode next = null;
        pre.next = null;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
    // 递归
    private ListNode reverse1(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverse1(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    private void mergeList(ListNode node1, ListNode node2) {
        ListNode next1 = null;
        ListNode next2 = null;
        while (node2 != null) {
            // 保存
            next1 = node1.next;
            next2 = node2.next;
            //　拼接
            node1.next = node2;
            node2.next = next1;
            // 移动指针
            node1 = next1;
            node2 = next2;
        }
    }

    // 147. Insertion Sort List
    // ****
    public ListNode insertionSortList(ListNode head) {
        ListNode helper = new ListNode(-1);
        ListNode newHead = helper;
        ListNode next = null;
        while (head != null) {
            // helper.next
            while (helper.next != null && helper.next.val < head.val) {
                helper = helper.next;
            }
            next = head.next;
            // 在helper和helper.next之间插入
            head.next = helper.next;
            helper.next = head;

            helper = newHead;
            head = next;
        }
        return newHead.next;
        
    }

    // 148. Sort List
    // ****
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode middle = getMiddle1(head);
        ListNode start = middle.next;
        middle.next = null;
        ListNode first = sortList(head);
        ListNode second = sortList(start);
        return merge(first, second);
    }

    public ListNode getMiddle1(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    private ListNode merge(ListNode first, ListNode second) {
        ListNode head = new ListNode(-1);
        ListNode helper = head;
        while (first != null && second != null) {
            if (first.val < second.val) {
                head.next = first;
                first = first.next;
            } else {
                head.next = second;
                second = second.next;
            }
            head = head.next;
        }

        head.next = first != null ? first : second;
        return helper.next;
    }


    // 2. Add Two Numbers
    // ***
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode newHead = new ListNode(-1);
        ListNode helper = newHead;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int value = carry;
            if (l1 != null) {
                value += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                value += l2.val;
                l2 = l2.next;
            }
            carry = value / 10;
            value = value % 10;
            ListNode cur = new ListNode(value);
            newHead.next = cur;
            newHead = newHead.next;
        }
        if (carry != 0) {
            newHead.next = new ListNode(carry);
        }
        return helper.next;
    }
    
}
