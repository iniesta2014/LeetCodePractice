package come.example.qiaowenhao.leetcodepractices;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class LeetCodeActivity extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leetcode);
        int[] nums = {2,5,6,0,0,1,2};
        //nextPermutation(nums);
        Log.d("iniesta", "onCreate: search " + search(nums, 3));
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
    // **** 双指针，理解双重循环
    public void sortColors(int[] nums) {
        int start = 0, end = nums.length - 1;
        for (int i = start; i <= end; i++) {
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
    //分析　https://blog.csdn.net/sinat_35261315/article/details/79205157
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

    // 86. Partition List　前半部分比x小，后半部分比x大
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
    // *****  没看懂
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

    //　迭代 记忆
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
    // **** 没看懂
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

    // 148. Sort List　无序变有序
    // *****
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

    public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
    }


    // 94. Binary Tree Inorder Traversal
    // *****
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else {
                TreeNode node = stack.pop();
                res.add(node.val);
                root = node.right;
            }
        }
        return res;
    }

    // 144. Binary Tree Preorder Traversal
    // *****
    public List<Integer> preorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                stack.push(root);
                res.add(root.val);
                root = root.left;
            } else {
                TreeNode node = stack.pop();
                root = node.right;
            }
        }
        return res;
    }

    // 145. Binary Tree Postorder Traversal
    //*****
    public List<Integer> postorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        LinkedList<Integer> res = new LinkedList<>();
        if (root == null) {
            return res;
        }
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                stack.push(root);
                res.addFirst(root.val);
                root = root.right;
            } else {
                TreeNode node = stack.pop();
                root = node.left;
            }
        }
        return res;
    }

    // 102. Binary Tree Level Order Traversal
    // ****关键是用队列存储一层
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root == null) {
            return res;
        }
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            res.add(tmp);
        }
        return res;
    }

    // 95. Unique Binary Search Trees II
    //*****
    public List<TreeNode> generateTrees(int n) {
        if (n < 1) {
            return new ArrayList();
        }
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if (start > end) {
            res.add(null);
            return res;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> lefts = generateTrees(start, i - 1);
            List<TreeNode> rights = generateTrees(i + 1, end);
            for (TreeNode left : lefts) {
                for (TreeNode right : rights) {
                    TreeNode node = new TreeNode(i);
                    node.left = left;
                    node.right = right;
                    res.add(node);
                }
            }
        }
        return res;
    }

    // 96. Unique Binary Search Trees
    // *****
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }

    // 98. Validate Binary Search Tree
    // 对每个结点保存上下界
    // ****
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean isValidBST(TreeNode root, long low, long high) {
        if (root == null) {
            return true;
        }
        if (root.val > low && root.val < high) {
            return isValidBST(root.left, low, root.val) && isValidBST(root.right, root.val, high);
        } else {
            return false;
        }
    }

    // 105. Construct Binary Tree from Preorder and Inorder Traversal
    // ****
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0) {
            return null;
        }
        int index = 0;
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == preorder[0]) {
                index = i;
                break;
            }
        }
        int[] inorderLeft = new int[index];
        int[] inorderRight = new int[inorder.length - index - 1];
        int[] preorderLeft = new int[index];
        int[] preorderRight = new int[preorder.length - index - 1];
        System.arraycopy(inorder,0, inorderLeft, 0, index);
        System.arraycopy(inorder, index + 1, inorderRight, 0, inorder.length - index - 1);
        System.arraycopy(preorder,1, preorderLeft, 0, index);
        System.arraycopy(preorder, index + 1, preorderRight, 0, preorder.length - index - 1);
        TreeNode root = new TreeNode(preorder[0]);
        root.left = buildTree(preorderLeft, inorderLeft);
        root.right = buildTree(preorderRight, inorderRight);
        return root;
    }

    // 113. Path Sum II
    // *****回溯
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        pathSum(res, tmp, root, sum);
        return res;
    }

    public void pathSum(List<List<Integer>> res, List<Integer> tmp, TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        tmp.add(root.val);

        if (root.left == null && root.right == null && root.val == sum) {
            res.add(new ArrayList<Integer>(tmp));
            // 因为和后面的条件互斥，不return也可以
        }
        if (root.left != null) {
            pathSum(res, tmp, root.left, sum - root.val);
            tmp.remove(tmp.size() - 1);
        }
        if (root.right != null) {
            pathSum(res, tmp, root.right, sum - root.val);
            tmp.remove(tmp.size() - 1);
        }
    }


    public class TreeLinkNode {
      int val;
      TreeLinkNode left, right, next;
      TreeLinkNode(int x) { val = x; }
  }

  // 114. Flatten Binary Tree to Linked List
  // ****
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            flatten(root.left);
        }
        if (root.right != null) {
            flatten(root.right);
        }

        TreeNode tmpRight = root.right;
        root.right = root.left;
        root.left = null;
        while (root.right != null) {
            root = root.right;
        }
        root.right = tmpRight;
    }

    // 116. Populating Next Right Pointers in Each Node
    // *****
    public void connect(TreeLinkNode root) {
        if (root == null) {
            return;
        }

        if (root.left != null) {
            root.left.next = root.right;
        }
        if (root.right != null) {
            if (root.next != null) {
                root.right.next = root.next.left;
            } else {
                root.right.next = null;
            }
        }

        connect(root.left);
        connect(root.right);
    }

    // 337. House Robber III
    // ****动态规划
    public int rob(TreeNode root) {
        int[] res = robHelper(root);
        return Math.max(res[0], res[1]);
    }

    private int[] robHelper(TreeNode root) {
        int[] dp = new int[2];
        if (root == null) {
            return dp;
        }
        int[] left = robHelper(root.left);
        int[] right = robHelper(root.right);
        dp[0] = root.val + left[1] + right[1];
        dp[1] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return dp;
    }

    // 129. Sum Root to Leaf Numbers
    // ****回溯
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }

        List<List<TreeNode>> res = new ArrayList<>();
        List<TreeNode> item = new ArrayList<>();
        dfs(root, res, item);
        int sum = 0;
        for (List<TreeNode> treeNodes : res) {
            StringBuilder sb = new StringBuilder();
            for (TreeNode treeNode : treeNodes) {
                sb.append(treeNode.val);
            }
            sum += Integer.valueOf(sb.toString());
        }
        return sum;
    }

    public void dfs(TreeNode root, List<List<TreeNode>> res, List<TreeNode> item) {
        item.add(root);
        if (root.left == null && root.right == null) {
            res.add(new ArrayList<TreeNode>(item));
        }
        if (root.left != null) {
            dfs(root.left, res, item);
            item.remove(item.size() - 1);
        }
        if (root.right != null) {
            dfs(root.right, res, item);
            item.remove(item.size() - 1);
        }
    }

    // 199. Binary Tree Right Side View
    // ****
    public List<Integer> rightSideView(TreeNode root) {
        Map<Integer, Integer> map = new HashMap<>();
        dfs(root, map, 0);
        List<Integer> res = new ArrayList<>();
        for (Integer item : map.values()) {
            res.add(item);
        }
        return res;
    }

    public void dfs(TreeNode root, Map<Integer, Integer> map, int level) {
        if (root == null) {
            return;
        }
        level++;
        map.put(level, root.val);
        if (root.left != null) {
            dfs(root.left, map, level);
        }
        if (root.right != null) {
            dfs(root.right, map, level);
        }
    }

    // 31. Next Permutation
    // *****
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int pos = -1;
        int length = nums.length;
        // 找到最后一个升序位置
        for (int i = length - 1; i >= 0; i--) {
            if (i - 1 >= 0 && nums[i] > nums[i - 1]) {
                pos = i -1;
                break;
            }
        }

        // 不存在升序位置，说明这个数是最大的，反排
        if (pos == -1) {
            reverse(nums, 0, length - 1);
            return;
        }

        // 找到pos之后最后比他大的位置，和pos交换
        for (int i = length - 1; i > pos; i--) {
            if (nums[i] > nums[pos]) {
                swapNum(nums, i, pos);
                break;
            }
        }

        // 反转pos之后的数
        reverse(nums, pos + 1, length - 1);
    }

    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            swapNum(nums, start, end);
            start++;
            end--;
        }
    }

    private void swapNum(int[] nums, int start, int end) {
        int tmp = nums[start];
        nums[start] = nums[end];
        nums[end] = tmp;
    }

    //  39. Combination Sum
    // ****
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        // 排序
        Arrays.sort(candidates);
        combinationSum(res, new ArrayList<Integer>(), candidates, target, 0);
        return res;
    }

    private void combinationSum(List<List<Integer>> res, List<Integer> tmp, int[] candidates, int target, int cur) {
        if (target < 0) {
            return;
        }

        // 注意错误写法
        //tmp.add(candidates[cur]);
        if (target == 0) {
            res.add(new ArrayList<Integer>(tmp));
        }

        // 遍历，每一步选哪一个都有可能
        for (int i = cur; i < candidates.length; i++) {
            tmp.add(candidates[i]);
            combinationSum(res, tmp, candidates, target - candidates[i], i);
            tmp.remove(tmp.size() - 1);
        }
    }

    // 48. Rotate Image
    // 图片顺时针旋转90
    // **** 注意值传递的问题，不能直接交换两个元素,数组一定要注意下标越界的错误
    public void rotate(int[][] matrix) {
        int row = matrix.length;
        int colum = matrix[0].length;

        for (int i = 0; i < row / 2; i++) {
            for (int j = 0; j < colum; j++) {
                swap(matrix, i, j, row - i - 1, j);
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < i; j++) {
                swap(matrix, i, j, j, i);
            }
        }

    }

    private void swap(int[][] matrix, int i, int j, int m, int k) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[m][k];
        matrix[m][k] = tmp;
    }

    // 54. Spiral Matrix m*n矩阵顺时针打印
    // *****
    // The only tricky part is that when I traverse left or up I have to check whether the row or
    // col still exists to prevent duplicates
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> list = new ArrayList<>();
        if (matrix == null) {
            return list;
        }
        int top = 0, bottom = matrix.length - 1;
        int start = 0, end = matrix[0].length - 1;
        // 1.是&&
        while (start <= end && top <= bottom) {
            for (int i = start; i <= end; i++) {
                list.add(matrix[top][i]);
            }
            top++;
            for (int i = top; i <= bottom; i++) {
                list.add(matrix[i][end]);
            }
            end--;

            // 2.判断越界与否
            if (top <= bottom) {
                for (int i = end; i >= start; i--) {
                    list.add(matrix[bottom][i]);
                }
                bottom--;
            }

            if (start <= end) {
                for (int i = bottom; i <= top; i--) {
                    list.add(matrix[i][start]);
                }
                start++;
            }
        }
        return list;
    }

    // 34. Find First and Last Position of Element in Sorted Array
    // 两次二分查找，第一次可能找不到
    // ***** mid取值，真正理解二分法
    public int[] searchRange(int[] nums, int target) {
        int[] res = {-1, -1};
        if (nums == null || nums.length == 0) {
            return res;
        }
        // search for the left one
        int start = 0;
        int end = nums.length - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (nums[mid] >= target) {
                end  = mid;
            } else {
                start = mid + 1;
            }
        }
        if (nums[start] == target) {
            res[0] = start;
        } else {
            return res;
        }
        // Search for the right one
        // We don't have to set i to 0 the second time.
        end = nums.length - 1;
        while (start < end) {
            int mid = (start + end) / 2 + 1;
            if (nums[mid] <= target) {
                start = mid;
            } else {
                end = mid - 1;
            }
        }
        // both start and end is ok
        res[1] = start;
        return res;
    }

    // 55. Jump Game
    // 最优解应该是贪心，这里先用动态规划，dp[i]表示到i剩余的跳数
    // 贪心解法 https://www.cnblogs.com/grandyang/p/4371526.html
    // ****
    public boolean canJump(int[] nums) {
        int length = nums.length;
        int[] dp = new int[length];
        for (int i = 1; i < length; i++) {
            dp[i] = Math.max(dp[i - 1], nums[i - 1]) - 1;
            if (dp[i] < 0) {
                return false;
            }
        }
        return true;
    }

    //  62. Unique Paths
    // *** 空间复杂度需要优化
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i < n; i++) {
            dp[0][i] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    // 63. Unique Paths II
    // **** dp数组的初始化
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }

        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) {
                dp[0][i] = 0;
                break;
            } else {
                dp[0][i] = 1;
            }
        }

        for (int j = 0; j < m; j++) {
            if (obstacleGrid[j][0] == 1) {
                dp[j][0] = 0;
                break;
            } else {
                dp[j][0] = 1;
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    // 74. Search a 2D Matrix
    // ***
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int m = matrix.length;
        int n = matrix[0].length;
        int start = 0;
        int end = m * n - 1;
        // 注意等号
        while (start <= end) {
            int mid = (start + end) / 2;
            int cur = matrix[mid / n][mid % n];
            if (cur == target) {
                return true;
            } else if (cur < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return false;
    }

    // 78. Subsets
    // *****
    // 理解回溯算法 需要结合解空间理解一下
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        dfs(res, new ArrayList<Integer>(), nums, 0);
        return res;
    }

    private void dfs(List<List<Integer>> res, List<Integer> tmp, int[] nums, int cur) {
        res.add(new ArrayList<Integer>(tmp));
        for (int i = cur; i < nums.length; i++) {
            tmp.add(nums[i]);
            dfs(res, tmp, nums, i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    // 另一种解法
    public List<List<Integer>> subsets1(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        List<Integer> array = new ArrayList<Integer>();
        result.add(array);
        if(nums==null){
            return result;
        }
        Arrays.sort(nums);
        for(int i=1;i<=nums.length;i++){
            array.clear();
            dfs1(nums,0,i,array,result);
        }
        return result;
    }
    void dfs1(int []number_array,int start,int number,List<Integer> array,
             List<List<Integer>> result) {
        if(number==array.size()){
            result.add(new ArrayList<Integer>(array)); //思考此处为何要新建一个数组？空数组是怎么得到的？
            return;
        }
        for(int i=start;i<number_array.length;i++){
            array.add(number_array[i]);
            dfs1(number_array,i+1,number,array,result);
            array.remove(array.size()-1);
        }
    }

    //  81. Search in Rotated Sorted Array II
    // 考察边界条件，如{1,3}
    // ******
    public boolean search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int start = 0;
        int end = nums.length - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) {
                return true;
            }
            // the only difference from the first one, trickly case, just updat left and right
            if (nums[start] == nums[mid] && nums[end] == nums[mid]) {
                start++;
                end--;
                // 左侧升序
            } else if (nums[mid] >= nums[start]) {
                if (nums[start] <= target && target < nums[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
                // 右侧升序
            } else {
                if (target > nums[mid] && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }
        return false;
    }

    // 238. Product of Array Except Self
    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] res = new int[length];
        int[] left = new int[length];
        int[] right = new int[length];
        left[0] = 1;
        right[length - 1] = 1;
        for (int i = 1; i < length; i++) {
            left[i] = left[i - 1] * nums[i - 1];
        }

        for (int j = length - 2; j >= 0; j--) {
            right[j] = right[j + 1] * nums[j + 1];
        }

        for (int k = 0; k < nums.length; k++) {
            res[k] = left[k] * right[k];
        }
        return res;
    }

    // 90. Subsets II
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        subsetsWithDup(res, new ArrayList<Integer>(), nums, 0);
        return res;
    }

    private void subsetsWithDup(List<List<Integer>> res, List<Integer> tmp, int[] nums, int cur) {
        res.add(new ArrayList<Integer>(tmp));
        for (int i = cur; i < nums.length; i++) {
            if (i > cur && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            subsetsWithDup(res, tmp, nums, i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    // 162. Find Peak Element
    // **** 还有一个避免复杂边界条件判断的方法
    public int findPeakElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int length = nums.length;
        int start = 0;
        int end = length - 1;
        int mid = 0;
        while (start <= end) {
            //在边界条件上非常容易出错
            mid = (start + end) / 2;
            if ((mid == 0 || nums[mid] > nums[mid - 1]) &&
                    (mid + 1 == length || nums[mid] > nums[mid + 1])) {
                return mid;
            } else if (mid > 0 && nums[mid] < nums[mid - 1]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return mid;
    }

    //  106. Construct Binary Tree from Inorder and Postorder Traversal
    public TreeNode buildTree1(int[] inorder, int[] postorder) {
        return buildTree(inorder, postorder, 0, inorder.length - 1, postorder.length - 1);
    }

    // 关键在于引入postEnd和递归终止条件
    private TreeNode buildTree(int[] inorder, int[] posterorder, int inStart, int inEnd, int postEnd) {
        if (inStart > inEnd) {
            return null;
        }
        int rootVal = posterorder[postEnd];
        int index = inStart;
        for (int i = index; i <= inEnd; i++) {
            if (inorder[i] == rootVal) {
                index = i;
            }
        }
        TreeNode root = new TreeNode(rootVal);
        root.left = buildTree(inorder, posterorder, inStart, index - 1, postEnd - 1 - (inEnd - index));
        root.right = buildTree(inorder, posterorder, index + 1, inEnd, postEnd - 1);
        return root;
    }

    // 64. Minimum Path Sum
    // ***
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length < 0) {
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        // 行列不一定相等
        for (int j = 1; j < m; j++) {
            dp[j][0] = dp[j - 1][0] + grid[j][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    // 216. Combination Sum III
    // ***
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (k < 0 || n < 0) {
            return res;
        }
        List<Integer> tmp = new ArrayList<>();
        combinationSum3(res, tmp, k, n, 1);
        return res;
    }

    private void combinationSum3(List<List<Integer>> res, List<Integer> tmp, int num, int sum, int cur) {
        // 注意, cur是可以等于10的
        if (num < 0 || sum < 0 || cur > 10) {
            return;
        }

        if (sum == 0 && num == 0) {
            res.add(new ArrayList<Integer>(tmp));
        }
        
        for (int i = cur; i <= 9; i++) {
            tmp.add(i);
            combinationSum3(res, tmp, num - 1, sum - i, i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    // 79.Word Search
    // 可以不使用额外的空间
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word == null || word.length() == 0) {
            return false;
        }
        int m = board.length;
        int n = board[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (exist(board, i, j, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean exist(char[][] board, int i, int j, int cur, String word) {
        if (cur == word.length()) {
            return true;
        }

        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || cur >= word.length()) {
            return false;
        }
        if (board[i][j] != word.charAt(cur)) {
            return false;
        }

        // board[y][x] ^= 256 it's a marker that the letter at position x,y is a part of word we search.
        //After board[y][x] ^= 256 the char became not a valid letter
        board[i][j] ^= 256;
        // 不要直接return
        boolean exist =  exist(board, i - 1, j, cur + 1, word) || exist(board, i + 1, j, cur + 1, word)
                || exist(board, i, j - 1, cur + 1, word) || exist(board, i, j + 1, cur + 1, word);
        board[i][j] ^= 256;
        return exist;
    }


    // 120. Triangle
    // *****自底向上，一维数组
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = triangle.get(n - 1).get(i);
        }
        for (int i = n - 2; i >= 0; i --) {
            for (int j = 0; j <= i; j++) {
                // 理解
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

    // 228. Summary Ranges
    // ****双重循环,可以理解为快慢指针
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        for (int i = 0; i < nums.length; i++) {
            int start = nums[i];
            while (i + 1 < nums.length && nums[i+1] - nums[i] == 1) {
                i++;
            }
            if (nums[i] > start) {
                res.add(start + "->" + nums[i]);
            } else {
                res.add(start + "");
            }
        }
        return res;
    }

    // 153. Find Minimum in Rotated Sorted Array
    // **** 结束条件，注意=(二分法一定要考虑只有一个或两个数这种情况)
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int start = 0;
        int end = nums.length - 1;
        while (start < end) {
            if (nums[start] < nums[end]) {
                return nums[start];
            }
            int mid = (start + end) / 2;
            if (nums[mid] >= nums[start]) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        return nums[start];
    }

    // 229. Majority Element II
    // ****　多数投票算法，第一次遍历确定candidate,第二次遍历判断是否满足次数要求
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        int candid1 = nums[0];
        int candid2 = nums[0];
        int count1 = 0;
        int count2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == candid1) {
                count1++;
            } else if (nums[i] == candid2) {
                count2++;
            } else if (count1 == 0) {
                candid1 = nums[i];
                count1++;
            } else if (count2 == 0) {
                candid2 = nums[i];
                count2++;
            } else {
                count1--;
                count2--;
            }
        }

        count1 = 0;
        count2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == candid1) {
                count1++;
            } else if (nums[i] == candid2) {
                count2++;
            }
        }
        if (count1 > nums.length / 3) {
            res.add(candid1);
        }
        if (count2 > nums.length / 3) {
            res.add(candid2);
        }
        return res;
    }

    // 289. Game of Life
    // ****　用两位二进制表示当前和下一刻的状态
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int colum = board[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < colum; j++) {
                int lives = getLives(board, i, j);
                if (board[i][j] == 1) {
                    if (lives == 2 || lives == 3) {
                        board[i][j] = 3;
                    }
                }
                if (board[i][j] == 0) {
                    if (lives == 3) {
                        board[i][j] = 2;
                    }
                }
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < colum; j++) {
                board[i][j] >>= 1;
            }
        }

    }

    private int getLives(int[][] board, int i, int j) {
        int res = 0;
        int row = board.length;
        int colum = board[0].length;
        for (int l = i - 1 ; l <= i + 1; l++) {
            if (l < 0 || l >= row - 1) {
                continue;
            }
            for (int m = j - 1; m <= j + 1; m++) {
                if (m < 0 || m >= colum - 1) {
                    continue;
                }
                if (l == i && m == j) {
                    continue;
                }
                if ((board[l][m] & 1) == 1) {
                    res++;
                }
            }
        }
        return res;
    }

    // 同getLives对比一下
    public int liveNeighbors(int[][] board, int m, int n, int i, int j) {
        int lives = 0;
        for (int x = Math.max(i - 1, 0); x <= Math.min(i + 1, m - 1); x++) {
            for (int y = Math.max(j - 1, 0); y <= Math.min(j + 1, n - 1); y++) {
                lives += board[x][y] & 1;
            }
        }
        lives -= board[i][j] & 1;
        return lives;
    }

}
